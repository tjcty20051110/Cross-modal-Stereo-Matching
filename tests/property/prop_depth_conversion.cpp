#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include "stereo/stereo_matcher.h"

RC_GTEST_PROP(DepthConversionProperty, FormulaAndClampingHold, (float disparity_raw, float depth_raw)) {
    const float disparity = std::fmod(std::abs(disparity_raw), 49.0f) + 0.01f;
    const float depth_limit = std::fmod(std::abs(depth_raw), 99.0f) + 0.1f;
    const double focal = 10.0;
    const double baseline = 0.2;

    cv::Mat disp(1, 1, CV_32F, cv::Scalar(disparity));
    cv::Mat mask(1, 1, CV_8U, cv::Scalar(255));
    auto result = cms::StereoMatcher::disparityToDepth(disp, mask, focal, baseline, depth_limit);

    const float expected = std::min(static_cast<float>((focal * baseline) / disparity), depth_limit);
    RC_ASSERT(result.valid_mask.at<unsigned char>(0, 0) == 255);
    RC_ASSERT(std::abs(result.depth_map.at<float>(0, 0) - expected) < 1e-5f);
}
