#include <cstdlib>
#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include "cost/cost_volume_builder.h"

RC_GTEST_PROP(CostVolumeProperty, ValidShapeProduced, (int h_raw, int w_raw, int d_raw)) {
    const int h = std::abs(h_raw) % 8 + 1;
    const int w = std::abs(w_raw) % 8 + 1;
    const int d = std::abs(d_raw) % w + 1;
    cms::PipelineConfig config;
    config.manifest_path = "manifest.csv";
    cms::FeatureTensor left;
    left.data = cv::Mat(h, w, CV_32FC1, cv::Scalar(1.0f));
    left.channels = 1;
    cms::FeatureTensor right = left;
    cms::CostVolumeBuilder builder(config);
    auto result = builder.build(left, right, d);
    RC_ASSERT(result.has_value());
    RC_ASSERT(result.value().height == h);
    RC_ASSERT(result.value().width == w);
    RC_ASSERT(result.value().max_disparity == d);
}

RC_GTEST_PROP(CostVolumeProperty, InvalidDisparityRejected, (int h_raw, int w_raw)) {
    const int h = std::abs(h_raw) % 8 + 1;
    const int w = std::abs(w_raw) % 8 + 1;
    cms::PipelineConfig config;
    config.manifest_path = "manifest.csv";
    cms::FeatureTensor left;
    left.data = cv::Mat(h, w, CV_32FC1, cv::Scalar(1.0f));
    left.channels = 1;
    cms::FeatureTensor right = left;
    cms::CostVolumeBuilder builder(config);
    auto result = builder.build(left, right, w + 1);
    RC_ASSERT(!result.has_value());
}
