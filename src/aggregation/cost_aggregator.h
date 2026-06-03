#pragma once

#include <memory>

#include <opencv2/core.hpp>

#include "common/types.h"
#include "config/config_manager.h"
#include "cost/cost_volume_builder.h"

namespace cms {

class ICostAggregator {
public:
    virtual ~ICostAggregator() = default;
    virtual Result<CostVolume> aggregate(
        const CostVolume& cost_volume,
        const cv::Mat& reference_image) const = 0;
};

class SGMAggregator : public ICostAggregator {
public:
    SGMAggregator(int p1, int p2, int num_directions = 8);
    Result<CostVolume> aggregate(
        const CostVolume& cost_volume,
        const cv::Mat& reference_image) const override;

private:
    int p1_;
    int p2_;
    int num_directions_;

    void aggregatePath(const CostVolume& cost,
                       CostVolume& aggregated,
                       const cv::Mat& ref_img,
                       int dx, int dy) const;
};

class GuidedFilterAggregator : public ICostAggregator {
public:
    GuidedFilterAggregator(int radius = 9, double eps = 0.01);
    Result<CostVolume> aggregate(
        const CostVolume& cost_volume,
        const cv::Mat& reference_image) const override;

private:
    int radius_;
    double eps_;
};

std::unique_ptr<ICostAggregator> createAggregator(const PipelineConfig& config);

}  // namespace cms
