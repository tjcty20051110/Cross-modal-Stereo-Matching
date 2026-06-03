#include <gtest/gtest.h>

#include "features/feature_extractor.h"

TEST(FeatureExtractorTest, CensusProducesExpectedChannels) {
    cms::CensusFeatureExtractor extractor(9);
    cv::Mat image(5, 6, CV_32F, cv::Scalar(0.5f));

    auto result = extractor.extract(image);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().channels, 80);
    EXPECT_EQ(result.value().downsample_factor, 1);
    EXPECT_EQ(result.value().data.rows, image.rows);
    EXPECT_EQ(result.value().data.cols, image.cols);
    EXPECT_EQ(result.value().data.channels(), 80);
}

TEST(FeatureExtractorTest, CensusOnConstantImageIsAllZero) {
    cms::CensusFeatureExtractor extractor(9);
    cv::Mat image(4, 4, CV_32F, cv::Scalar(1.0f));

    auto result = extractor.extract(image);
    ASSERT_TRUE(result.has_value());
    std::vector<cv::Mat> channels;
    cv::split(result.value().data, channels);
    for (const auto& ch : channels) {
        EXPECT_EQ(cv::countNonZero(ch), 0);
    }
}
