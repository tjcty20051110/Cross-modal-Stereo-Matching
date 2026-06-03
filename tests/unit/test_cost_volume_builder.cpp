#include <gtest/gtest.h>

#include "cost/cost_volume_builder.h"

TEST(CostVolumeBuilderTest, BuildsExpectedShape) {
    cms::PipelineConfig config;
    config.manifest_path = "manifest.csv";
    config.cost_metric = cms::CostMetric::CENSUS_HAMMING;

    cms::FeatureTensor left;
    left.data = cv::Mat(3, 4, CV_32FC2, cv::Scalar(0, 1));
    left.channels = 2;
    left.downsample_factor = 1;

    cms::FeatureTensor right = left;
    cms::CostVolumeBuilder builder(config);
    auto result = builder.build(left, right, 4);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().height, 3);
    EXPECT_EQ(result.value().width, 4);
    EXPECT_EQ(result.value().max_disparity, 4);
    EXPECT_EQ(result.value().data.rows, 12);
    EXPECT_EQ(result.value().data.cols, 4);
}

TEST(CostVolumeBuilderTest, RejectsInvalidMaxDisparity) {
    cms::PipelineConfig config;
    config.manifest_path = "manifest.csv";
    cms::FeatureTensor left;
    left.data = cv::Mat(2, 3, CV_32FC1, cv::Scalar(1));
    left.channels = 1;
    left.downsample_factor = 1;
    cms::FeatureTensor right = left;

    cms::CostVolumeBuilder builder(config);
    auto result = builder.build(left, right, 0);
    EXPECT_FALSE(result.has_value());
}

TEST(CostVolumeBuilderTest, SymmetricBuildReturnsTwoVolumes) {
    cms::PipelineConfig config;
    config.manifest_path = "manifest.csv";
    cms::FeatureTensor left;
    left.data = cv::Mat(2, 3, CV_32FC1, cv::Scalar(1));
    left.channels = 1;
    left.downsample_factor = 1;
    cms::FeatureTensor right = left;

    cms::CostVolumeBuilder builder(config);
    auto result = builder.buildSymmetric(left, right, 2);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().first.height, 2);
    EXPECT_EQ(result.value().second.height, 2);
    EXPECT_EQ(result.value().first.max_disparity, 2);
    EXPECT_EQ(result.value().second.max_disparity, 2);
}
