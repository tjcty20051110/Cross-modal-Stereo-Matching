#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include "stereo/stereo_matcher.h"

RC_GTEST_PROP(NumericalStabilityProperty, ConstantInputsProduceFiniteDisparity, (int rows_raw, int cols_raw)) {
    const int rows = std::abs(rows_raw) % 24 + 1;
    const int cols = std::abs(cols_raw) % 24 + 1;
    cms::PipelineConfig config;
    config.manifest_path = "manifest.csv";
    config.max_disparity = std::max(1, std::min(8, cols));

    cms::StereoSample sample;
    sample.left_image = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(20, 20, 20));
    sample.right_image = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(20));
    sample.left_modality = cms::Modality::RGB;
    sample.right_modality = cms::Modality::NIR;
    sample.valid_mask = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));
    sample.sample_id = "stable";

    cms::StereoMatcher matcher(config);
    auto result = matcher.match(sample);
    RC_ASSERT(result.has_value());
    for (int y = 0; y < result.value().disparity.rows; ++y) {
        for (int x = 0; x < result.value().disparity.cols; ++x) {
            RC_ASSERT(std::isfinite(result.value().disparity.at<float>(y, x)));
        }
    }
}
