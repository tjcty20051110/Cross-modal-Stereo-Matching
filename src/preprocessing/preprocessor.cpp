#include "preprocessing/preprocessor.h"

#include <algorithm>

#include <opencv2/imgproc.hpp>

namespace cms {

Preprocessor::Preprocessor(const PipelineConfig& config) : config_(config) {}

Result<PreprocessedPair> Preprocessor::process(
    const cv::Mat& left_image,
    const cv::Mat& right_image,
    Modality left_modality,
    Modality right_modality) const {
    if (left_image.empty() || right_image.empty()) {
        return Result<PreprocessedPair>::error("Preprocessor received empty image");
    }

    // Handle resolution mismatch for cross-modal pairs by resizing right to match left
    cv::Mat right_resized;
    if (left_image.size() != right_image.size()) {
        cv::resize(right_image, right_resized, left_image.size(), 0, 0, cv::INTER_LINEAR);
    } else {
        right_resized = right_image;
    }

    PreprocessedPair pair;
    pair.left = toNormalizedGray(left_image, left_modality);
    pair.right = toNormalizedGray(right_resized, right_modality);

    if (config_.enable_clahe) {
        pair.left = applyCLAHE(pair.left);
        pair.right = applyCLAHE(pair.right);
    }

    if (config_.enable_gradient_alignment) {
        pair.left_gradient = computeGradient(pair.left);
        pair.right_gradient = computeGradient(pair.right);
    }

    return Result<PreprocessedPair>::success(pair);
}

cv::Mat Preprocessor::toNormalizedGray(const cv::Mat& img, Modality /*modality*/) const {
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = img.clone();
    }

    cv::Mat gray32;
    if (gray.depth() == CV_8U) {
        gray.convertTo(gray32, CV_32F, 1.0 / 255.0);
    } else if (gray.depth() == CV_16U) {
        gray.convertTo(gray32, CV_32F, 1.0 / 65535.0);
    } else {
        gray.convertTo(gray32, CV_32F);
        double min_v = 0.0;
        double max_v = 0.0;
        cv::minMaxLoc(gray32, &min_v, &max_v);
        if (min_v < 0.0) {
            gray32 -= static_cast<float>(min_v);
            max_v -= min_v;
        }
        if (max_v > 1.0 + EPSILON) {
            gray32 /= static_cast<float>(max_v + EPSILON);
        }
    }

    cv::threshold(gray32, gray32, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::max(gray32, 0.0, gray32);
    return gray32;
}

cv::Mat Preprocessor::applyCLAHE(const cv::Mat& img) const {
    cv::Mat input8u;
    img.convertTo(input8u, CV_8U, 255.0);
    auto clahe = cv::createCLAHE(static_cast<double>(config_.clahe_clip_limit),
                                 cv::Size(config_.clahe_tile_size, config_.clahe_tile_size));
    cv::Mat enhanced8u;
    clahe->apply(input8u, enhanced8u);
    cv::Mat enhanced32f;
    enhanced8u.convertTo(enhanced32f, CV_32F, 1.0 / 255.0);
    return enhanced32f;
}

cv::Mat Preprocessor::computeGradient(const cv::Mat& img) const {
    cv::Mat gx;
    cv::Mat gy;
    cv::Sobel(img, gx, CV_32F, 1, 0, 3);
    cv::Sobel(img, gy, CV_32F, 0, 1, 3);
    cv::Mat magnitude;
    cv::magnitude(gx, gy, magnitude);

    double min_v = 0.0;
    double max_v = 0.0;
    cv::minMaxLoc(magnitude, &min_v, &max_v);
    if (max_v > min_v + EPSILON) {
        magnitude = (magnitude - static_cast<float>(min_v)) / static_cast<float>(max_v - min_v + EPSILON);
    } else {
        magnitude = cv::Mat::zeros(magnitude.size(), CV_32F);
    }
    return magnitude;
}

}  // namespace cms
