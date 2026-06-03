#include "stereo/stereo_matcher.h"

namespace cms {

StereoMatcher::StereoMatcher(const PipelineConfig& config)
    : config_(config),
      preprocessor_(std::make_unique<Preprocessor>(config)),
      feature_extractor_(createFeatureExtractor(config)),
      cost_volume_builder_(std::make_unique<CostVolumeBuilder>(config)),
      cost_aggregator_(createAggregator(config)),
      disparity_estimator_(std::make_unique<DisparityEstimator>(config)) {}

Result<DisparityResult> StereoMatcher::match(const StereoSample& sample) {
    auto preprocessed = preprocessor_->process(
        sample.left_image, sample.right_image, sample.left_modality, sample.right_modality);
    if (!preprocessed) return Result<DisparityResult>::error(preprocessed.error_msg());

    auto left_feat = feature_extractor_->extract(preprocessed.value().left);
    if (!left_feat) return Result<DisparityResult>::error(left_feat.error_msg());
    auto right_feat = feature_extractor_->extract(preprocessed.value().right);
    if (!right_feat) return Result<DisparityResult>::error(right_feat.error_msg());

    if (config_.enable_lr_check) {
        auto pair = cost_volume_builder_->buildSymmetric(left_feat.value(), right_feat.value(), config_.max_disparity);
        if (!pair) return Result<DisparityResult>::error(pair.error_msg());
        auto left_aggr = cost_aggregator_->aggregate(pair.value().first, preprocessed.value().left);
        if (!left_aggr) return Result<DisparityResult>::error(left_aggr.error_msg());
        auto right_aggr = cost_aggregator_->aggregate(pair.value().second, preprocessed.value().right);
        if (!right_aggr) return Result<DisparityResult>::error(right_aggr.error_msg());
        return disparity_estimator_->estimateWithLRCheck(
            left_aggr.value(), right_aggr.value(), left_feat.value().downsample_factor);
    }

    auto cost = cost_volume_builder_->build(left_feat.value(), right_feat.value(), config_.max_disparity);
    if (!cost) return Result<DisparityResult>::error(cost.error_msg());
    auto aggregated = cost_aggregator_->aggregate(cost.value(), preprocessed.value().left);
    if (!aggregated) return Result<DisparityResult>::error(aggregated.error_msg());
    return disparity_estimator_->estimate(aggregated.value(), left_feat.value().downsample_factor);
}

DepthResult StereoMatcher::disparityToDepth(
    const cv::Mat& disparity,
    const cv::Mat& confidence_mask,
    double focal_length,
    double baseline,
    double depth_max_meter) {
    DepthResult result;
    result.depth_map = cv::Mat(disparity.size(), CV_32F, cv::Scalar(0));
    result.valid_mask = confidence_mask.clone();

    for (int y = 0; y < disparity.rows; ++y) {
        for (int x = 0; x < disparity.cols; ++x) {
            const float d = disparity.at<float>(y, x);
            if (confidence_mask.at<unsigned char>(y, x) == 0 || d <= 0.0f) {
                result.valid_mask.at<unsigned char>(y, x) = 0;
                continue;
            }
            float depth = static_cast<float>(safeDivide(
                focal_length * baseline, static_cast<double>(d)));
            if (depth > depth_max_meter) depth = static_cast<float>(depth_max_meter);
            result.depth_map.at<float>(y, x) = depth;
        }
    }
    return result;
}

}  // namespace cms
