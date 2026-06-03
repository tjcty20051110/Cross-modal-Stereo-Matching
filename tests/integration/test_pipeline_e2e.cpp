#include <cmath>

#include <gtest/gtest.h>

#include "stereo/stereo_matcher.h"

TEST(PipelineE2ETest, MatcherProducesFiniteDisparityMapOnSyntheticSample) {
    cms::PipelineConfig config;
    config.manifest_path = "manifest.csv";
    config.max_disparity = 8;
    config.sgm_num_directions = 4;

    cms::StereoSample sample;
    sample.left_image = cv::Mat(32, 32, CV_8UC3);
    sample.right_image = cv::Mat(32, 32, CV_8UC1);
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
            const unsigned char value = static_cast<unsigned char>((x * 8 + y * 3) % 255);
            sample.left_image.at<cv::Vec3b>(y, x) = cv::Vec3b(value, value, value);
            sample.right_image.at<unsigned char>(y, x) = value;
        }
    }
    sample.left_modality = cms::Modality::RGB;
    sample.right_modality = cms::Modality::NIR;
    sample.valid_mask = cv::Mat(32, 32, CV_8U, cv::Scalar(255));
    sample.sample_id = "e2e";

    cms::StereoMatcher matcher(config);
    auto result = matcher.match(sample);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().disparity.rows, 32);
    EXPECT_EQ(result.value().disparity.cols, 32);
    for (int y = 0; y < result.value().disparity.rows; ++y) {
        for (int x = 0; x < result.value().disparity.cols; ++x) {
            EXPECT_TRUE(std::isfinite(result.value().disparity.at<float>(y, x)));
        }
    }
}
