#pragma once

#include <opencv2/core.hpp>

#include "common/types.h"
#include "config/config_manager.h"
#include "data/data_loader.h"

namespace cms {

struct PreprocessedPair {
    cv::Mat left;
    cv::Mat right;
    cv::Mat left_gradient;
    cv::Mat right_gradient;
};

class Preprocessor {
public:
    explicit Preprocessor(const PipelineConfig& config);

    Result<PreprocessedPair> process(
        const cv::Mat& left_image,
        const cv::Mat& right_image,
        Modality left_modality,
        Modality right_modality) const;

private:
    PipelineConfig config_;

    cv::Mat toNormalizedGray(const cv::Mat& img, Modality modality) const;
    cv::Mat applyCLAHE(const cv::Mat& img) const;
    cv::Mat computeGradient(const cv::Mat& img) const;
};

}  // namespace cms
