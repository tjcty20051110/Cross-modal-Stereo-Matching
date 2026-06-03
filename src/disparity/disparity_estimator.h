#pragma once

#include <opencv2/core.hpp>

#include "common/types.h"
#include "config/config_manager.h"
#include "cost/cost_volume_builder.h"

namespace cms {

struct DisparityResult {
    cv::Mat disparity;
    cv::Mat confidence_mask;
    int original_height = 0;
    int original_width = 0;
};

class DisparityEstimator {
public:
    explicit DisparityEstimator(const PipelineConfig& config);

    Result<DisparityResult> estimate(
        const CostVolume& aggregated_cost,
        int downsample_factor = 1) const;

    Result<DisparityResult> estimateWithLRCheck(
        const CostVolume& left_cost,
        const CostVolume& right_cost,
        int downsample_factor = 1) const;

private:
    PipelineConfig config_;

    cv::Mat winnerTakeAll(const CostVolume& cost) const;
    void subpixelRefine(cv::Mat& disparity, const CostVolume& cost) const;
    cv::Mat lrConsistencyCheck(const cv::Mat& left_disp,
                               const cv::Mat& right_disp) const;
    void fillHoles(cv::Mat& disparity, const cv::Mat& mask) const;
    void sanitize(cv::Mat& disparity, cv::Mat& mask) const;
};

}  // namespace cms
