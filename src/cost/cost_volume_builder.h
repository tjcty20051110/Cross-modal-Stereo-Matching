#pragma once

#include <utility>

#include <opencv2/core.hpp>

#include "common/types.h"
#include "config/config_manager.h"
#include "features/feature_extractor.h"

namespace cms {

struct CostVolume {
    cv::Mat data;
    int height = 0;
    int width = 0;
    int max_disparity = 0;

    float& at(int y, int x, int d);
    float at(int y, int x, int d) const;
};

class CostVolumeBuilder {
public:
    explicit CostVolumeBuilder(const PipelineConfig& config);

    Result<CostVolume> build(
        const FeatureTensor& left_feat,
        const FeatureTensor& right_feat,
        int max_disparity) const;

    Result<std::pair<CostVolume, CostVolume>> buildSymmetric(
        const FeatureTensor& left_feat,
        const FeatureTensor& right_feat,
        int max_disparity) const;

private:
    PipelineConfig config_;

    float computeCensusHamming(const cv::Mat& left, const cv::Mat& right,
                               int y, int x_left, int x_right) const;
    float computeNCC(const cv::Mat& left, const cv::Mat& right,
                     int y, int x_left, int x_right) const;
    float computeCosineSimilarity(const cv::Mat& left, const cv::Mat& right,
                                  int y, int x_left, int x_right) const;
    float computeCost(const cv::Mat& left, const cv::Mat& right,
                      int y, int x_left, int x_right) const;
};

}  // namespace cms
