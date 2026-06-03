#include "disparity/disparity_estimator.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace cms {

DisparityEstimator::DisparityEstimator(const PipelineConfig& config) : config_(config) {}

Result<DisparityResult> DisparityEstimator::estimate(
    const CostVolume& aggregated_cost,
    int downsample_factor) const {
    cv::Mat disparity = winnerTakeAll(aggregated_cost);
    subpixelRefine(disparity, aggregated_cost);
    cv::Mat mask(disparity.size(), CV_8U, cv::Scalar(255));
    sanitize(disparity, mask);

    if (downsample_factor > 1) {
        cv::resize(disparity, disparity, cv::Size(), downsample_factor, downsample_factor, cv::INTER_LINEAR);
        disparity *= static_cast<float>(downsample_factor);
        cv::resize(mask, mask, disparity.size(), 0, 0, cv::INTER_NEAREST);
    }

    DisparityResult result;
    result.disparity = disparity;
    result.confidence_mask = mask;
    result.original_height = disparity.rows;
    result.original_width = disparity.cols;
    return Result<DisparityResult>::success(result);
}

Result<DisparityResult> DisparityEstimator::estimateWithLRCheck(
    const CostVolume& left_cost,
    const CostVolume& right_cost,
    int downsample_factor) const {
    cv::Mat left_disp = winnerTakeAll(left_cost);
    cv::Mat right_disp = winnerTakeAll(right_cost);
    subpixelRefine(left_disp, left_cost);
    subpixelRefine(right_disp, right_cost);

    cv::Mat mask = lrConsistencyCheck(left_disp, right_disp);
    if (config_.enable_hole_filling) {
        fillHoles(left_disp, mask);
    }
    sanitize(left_disp, mask);

    if (downsample_factor > 1) {
        cv::resize(left_disp, left_disp, cv::Size(), downsample_factor, downsample_factor, cv::INTER_LINEAR);
        left_disp *= static_cast<float>(downsample_factor);
        cv::resize(mask, mask, left_disp.size(), 0, 0, cv::INTER_NEAREST);
    }

    DisparityResult result;
    result.disparity = left_disp;
    result.confidence_mask = mask;
    result.original_height = left_disp.rows;
    result.original_width = left_disp.cols;
    return Result<DisparityResult>::success(result);
}

cv::Mat DisparityEstimator::winnerTakeAll(const CostVolume& cost) const {
    cv::Mat disparity(cost.height, cost.width, CV_32F, cv::Scalar(0));
    for (int y = 0; y < cost.height; ++y) {
        for (int x = 0; x < cost.width; ++x) {
            float best_cost = cost.at(y, x, 0);
            int best_d = 0;
            for (int d = 1; d < cost.max_disparity; ++d) {
                float value = cost.at(y, x, d);
                if (value < best_cost) {
                    best_cost = value;
                    best_d = d;
                }
            }
            disparity.at<float>(y, x) = static_cast<float>(best_d);
        }
    }
    return disparity;
}

void DisparityEstimator::subpixelRefine(cv::Mat& disparity, const CostVolume& cost) const {
    for (int y = 0; y < cost.height; ++y) {
        for (int x = 0; x < cost.width; ++x) {
            const int d = static_cast<int>(std::round(disparity.at<float>(y, x)));
            if (d <= 0 || d >= cost.max_disparity - 1) {
                continue;
            }
            const float c1 = cost.at(y, x, d - 1);
            const float c2 = cost.at(y, x, d);
            const float c3 = cost.at(y, x, d + 1);
            const float denom = 2.0f * (c1 - 2.0f * c2 + c3);
            if (std::abs(denom) <= static_cast<float>(EPSILON)) {
                continue;
            }
            float delta = (c1 - c3) / denom;
            delta = std::max(-1.0f, std::min(1.0f, delta));
            disparity.at<float>(y, x) = static_cast<float>(d) + delta;
        }
    }
}

cv::Mat DisparityEstimator::lrConsistencyCheck(const cv::Mat& left_disp,
                                               const cv::Mat& right_disp) const {
    cv::Mat mask(left_disp.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < left_disp.rows; ++y) {
        for (int x = 0; x < left_disp.cols; ++x) {
            const float dl = left_disp.at<float>(y, x);
            const int xr = x - static_cast<int>(std::round(dl));
            if (xr < 0 || xr >= right_disp.cols) continue;
            const float dr = right_disp.at<float>(y, xr);
            if (std::abs(dl - dr) <= static_cast<float>(config_.lr_threshold)) {
                mask.at<unsigned char>(y, x) = 255;
            }
        }
    }
    return mask;
}

void DisparityEstimator::fillHoles(cv::Mat& disparity, const cv::Mat& mask) const {
    for (int y = 0; y < disparity.rows; ++y) {
        for (int x = 0; x < disparity.cols; ++x) {
            if (mask.at<unsigned char>(y, x) != 0) continue;
            float left_value = -1.0f;
            float right_value = -1.0f;
            for (int xl = x - 1; xl >= 0; --xl) {
                if (mask.at<unsigned char>(y, xl) != 0) {
                    left_value = disparity.at<float>(y, xl);
                    break;
                }
            }
            for (int xr = x + 1; xr < disparity.cols; ++xr) {
                if (mask.at<unsigned char>(y, xr) != 0) {
                    right_value = disparity.at<float>(y, xr);
                    break;
                }
            }
            if (left_value >= 0.0f && right_value >= 0.0f) {
                disparity.at<float>(y, x) = 0.5f * (left_value + right_value);
            } else if (left_value >= 0.0f) {
                disparity.at<float>(y, x) = left_value;
            } else if (right_value >= 0.0f) {
                disparity.at<float>(y, x) = right_value;
            } else {
                disparity.at<float>(y, x) = 0.0f;
            }
        }
    }
}

void DisparityEstimator::sanitize(cv::Mat& disparity, cv::Mat& mask) const {
    for (int y = 0; y < disparity.rows; ++y) {
        for (int x = 0; x < disparity.cols; ++x) {
            float& value = disparity.at<float>(y, x);
            if (!std::isfinite(value) || value < 0.0f) {
                value = 0.0f;
                mask.at<unsigned char>(y, x) = 0;
            }
        }
    }
}

}  // namespace cms
