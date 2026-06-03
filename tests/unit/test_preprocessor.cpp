#include <gtest/gtest.h>

#include "preprocessing/preprocessor.h"

TEST(PreprocessorTest, ConvertsRgbAndKeepsRange) {
    cms::PipelineConfig config;
    config.manifest_path = "manifest.csv";
    config.enable_clahe = false;
    cms::Preprocessor preprocessor(config);

    cv::Mat left(4, 4, CV_8UC3, cv::Scalar(10, 20, 30));
    cv::Mat right(4, 4, CV_16UC1, cv::Scalar(1000));
    auto result = preprocessor.process(left, right, cms::Modality::RGB, cms::Modality::NIR);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().left.type(), CV_32F);
    EXPECT_EQ(result.value().right.type(), CV_32F);
    double min_v = 0.0, max_v = 0.0;
    cv::minMaxLoc(result.value().left, &min_v, &max_v);
    EXPECT_GE(min_v, 0.0);
    EXPECT_LE(max_v, 1.0);
}
