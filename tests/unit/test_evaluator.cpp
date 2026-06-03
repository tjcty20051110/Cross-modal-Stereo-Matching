#include <cmath>

#include <gtest/gtest.h>

#include "evaluation/evaluator.h"

TEST(EvaluatorTest, ComputesMetricsOnValidPixelsOnly) {
    cv::Mat pred = (cv::Mat_<float>(1, 3) << 1.0f, 10.0f, 3.0f);
    cv::Mat gt = (cv::Mat_<float>(1, 3) << 1.0f, 2.0f, 1.0f);
    cv::Mat mask = (cv::Mat_<unsigned char>(1, 3) << 255, 0, 255);

    auto metrics = cms::Evaluator::evaluate(pred, gt, mask, "sample");
    EXPECT_TRUE(metrics.is_valid);
    EXPECT_EQ(metrics.valid_pixel_count, 2);
    EXPECT_NEAR(metrics.mse, 2.0, 1e-6);
    EXPECT_NEAR(metrics.epe, 1.0, 1e-6);
}

TEST(EvaluatorTest, ReturnsNaNWhenNoValidPixels) {
    cv::Mat pred = cv::Mat::zeros(2, 2, CV_32F);
    cv::Mat gt = cv::Mat::zeros(2, 2, CV_32F);
    cv::Mat mask = cv::Mat::zeros(2, 2, CV_8U);

    auto metrics = cms::Evaluator::evaluate(pred, gt, mask);
    EXPECT_FALSE(metrics.is_valid);
    EXPECT_TRUE(std::isnan(metrics.mse));
    EXPECT_TRUE(std::isnan(metrics.epe));
    EXPECT_TRUE(std::isnan(metrics.d1_all));
}
