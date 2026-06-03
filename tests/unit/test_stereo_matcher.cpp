#include <gtest/gtest.h>

#include "stereo/stereo_matcher.h"

TEST(StereoMatcherTest, DisparityToDepthAppliesFormulaAndMasking) {
    cv::Mat disparity = (cv::Mat_<float>(1, 3) << 2.0f, 0.0f, 0.5f);
    cv::Mat mask = (cv::Mat_<unsigned char>(1, 3) << 255, 255, 255);

    auto result = cms::StereoMatcher::disparityToDepth(disparity, mask, 10.0, 0.2, 1.0);
    EXPECT_NEAR(result.depth_map.at<float>(0, 0), 1.0f, 1e-6f);
    EXPECT_EQ(result.depth_map.at<float>(0, 1), 0.0f);
    EXPECT_EQ(result.valid_mask.at<unsigned char>(0, 1), 0);
    EXPECT_NEAR(result.depth_map.at<float>(0, 2), 1.0f, 1e-6f);
}
