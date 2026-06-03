#pragma once

#include <memory>

#include <opencv2/core.hpp>

#include "aggregation/cost_aggregator.h"
#include "common/types.h"
#include "config/config_manager.h"
#include "cost/cost_volume_builder.h"
#include "data/data_loader.h"
#include "disparity/disparity_estimator.h"
#include "features/feature_extractor.h"
#include "preprocessing/preprocessor.h"

namespace cms {

struct DepthResult {
    cv::Mat depth_map;
    cv::Mat valid_mask;
};

class StereoMatcher {
public:
    explicit StereoMatcher(const PipelineConfig& config);
    Result<DisparityResult> match(const StereoSample& sample);

    static DepthResult disparityToDepth(
        const cv::Mat& disparity,
        const cv::Mat& confidence_mask,
        double focal_length,
        double baseline,
        double depth_max_meter);

private:
    PipelineConfig config_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<IFeatureExtractor> feature_extractor_;
    std::unique_ptr<CostVolumeBuilder> cost_volume_builder_;
    std::unique_ptr<ICostAggregator> cost_aggregator_;
    std::unique_ptr<DisparityEstimator> disparity_estimator_;
};

}  // namespace cms
