#include "visualization/visualizer.h"

#include <algorithm>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace cms {
namespace {
int toOpenCVColorMap(ColorMap cmap) {
    switch (cmap) {
        case ColorMap::JET: return cv::COLORMAP_JET;
        case ColorMap::MAGMA: return cv::COLORMAP_INFERNO;
        case ColorMap::TURBO:
        default: return cv::COLORMAP_TURBO;
    }
}

cv::Mat toDisplayBGR(const cv::Mat& image) {
    cv::Mat out;
    if (image.empty()) return out;
    cv::Mat norm;
    if (image.depth() == CV_8U) {
        norm = image.clone();
    } else {
        double min_v = 0.0;
        double max_v = 0.0;
        cv::minMaxLoc(image, &min_v, &max_v);
        if (max_v <= min_v) {
            norm = cv::Mat(image.size(), CV_8U, cv::Scalar(0));
        } else {
            image.convertTo(norm, CV_8U, 255.0 / (max_v - min_v), -min_v * 255.0 / (max_v - min_v));
        }
    }
    if (norm.channels() == 1) {
        cv::cvtColor(norm, out, cv::COLOR_GRAY2BGR);
    } else if (norm.channels() == 4) {
        cv::cvtColor(norm, out, cv::COLOR_BGRA2BGR);
    } else {
        out = norm;
    }
    return out;
}

void putTitle(cv::Mat& image, const std::string& title) {
    cv::putText(image, title, cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}
}  // namespace

Visualizer::Visualizer(const PipelineConfig& config) : config_(config) {}

cv::Mat Visualizer::renderDisparityColor(const cv::Mat& disparity,
                                         double max_disp,
                                         ColorMap cmap) const {
    cv::Mat normalized;
    disparity.convertTo(normalized, CV_8U, 255.0 / std::max(max_disp, 1.0));
    cv::Mat colored;
    cv::applyColorMap(normalized, colored, toOpenCVColorMap(cmap));
    return colored;
}

cv::Mat Visualizer::renderErrorHeatmap(const cv::Mat& pred,
                                       const cv::Mat& gt,
                                       const cv::Mat& valid_mask) const {
    cv::Mat error = cv::Mat::zeros(pred.size(), CV_32F);
    for (int y = 0; y < pred.rows; ++y) {
        for (int x = 0; x < pred.cols; ++x) {
            if (valid_mask.at<unsigned char>(y, x) == 0) continue;
            error.at<float>(y, x) = std::abs(pred.at<float>(y, x) - gt.at<float>(y, x));
        }
    }
    cv::Mat heatmap = renderDisparityColor(error, 10.0, ColorMap::JET);
    cv::Mat contour_mask = error > 3.0f;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(contour_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(heatmap, contours, -1, cv::Scalar(255, 255, 255), 1);
    return heatmap;
}

cv::Mat Visualizer::renderComparison(const cv::Mat& left_rgb,
                                     const cv::Mat& right_ir,
                                     const cv::Mat& pred_color,
                                     const cv::Mat& gt_color,
                                     bool gt_available) const {
    cv::Mat left = toDisplayBGR(left_rgb);
    cv::Mat right = toDisplayBGR(right_ir);
    cv::Mat pred = toDisplayBGR(pred_color);
    cv::Mat gt = gt_available ? toDisplayBGR(gt_color) : cv::Mat(pred.size(), CV_8UC3, cv::Scalar(127, 127, 127));

    cv::resize(right, right, left.size());
    cv::resize(pred, pred, left.size());
    cv::resize(gt, gt, left.size());

    putTitle(left, "Left RGB");
    putTitle(right, "Right IR");
    putTitle(pred, "Prediction");
    putTitle(gt, gt_available ? "Ground Truth" : "GT unavailable");

    cv::Mat top, bottom, grid;
    cv::hconcat(std::vector<cv::Mat>{left, right}, top);
    cv::hconcat(std::vector<cv::Mat>{pred, gt}, bottom);
    cv::vconcat(std::vector<cv::Mat>{top, bottom}, grid);
    return grid;
}

void Visualizer::save(const cv::Mat& image,
                      const std::string& sample_id,
                      const std::string& suffix,
                      const std::filesystem::path& output_dir) const {
    std::filesystem::create_directories(output_dir);
    const auto out = output_dir / (sample_id + "_" + suffix + ".png");
    cv::imwrite(out.string(), image);
}

}  // namespace cms
