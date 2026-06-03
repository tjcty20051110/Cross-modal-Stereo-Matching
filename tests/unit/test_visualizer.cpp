#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "visualization/visualizer.h"
#include "config/config_manager.h"

using namespace cms;

TEST(VisualizerTest, SparseDisparityPointsAreExpandedForReadability) {
    PipelineConfig config;
    config.max_disparity = 128.0;
    Visualizer viz(config);

    // Create a very sparse disparity map: only a few valid points
    cv::Mat sparse_disp(100, 200, CV_32F, cv::Scalar(0.0f));
    sparse_disp.at<float>(50, 50) = 64.0f;
    sparse_disp.at<float>(50, 51) = 64.0f;
    sparse_disp.at<float>(51, 50) = 64.0f;
    sparse_disp.at<float>(51, 51) = 64.0f;

    auto colored = viz.renderDisparityColor(sparse_disp, 128.0, ColorMap::TURBO);

    EXPECT_EQ(colored.rows, 100);
    EXPECT_EQ(colored.cols, 200);
    EXPECT_EQ(colored.channels(), 3);

    // Center valid pixels must be colored
    cv::Vec3b black(0, 0, 0);
    cv::Vec3b center_color = colored.at<cv::Vec3b>(50, 50);
    EXPECT_NE(center_color, black);

    // Nearby pixels should also be non-black due to sparse-data dilation (3x3 ellipse)
    cv::Vec3b nearby = colored.at<cv::Vec3b>(50, 52);
    EXPECT_NE(nearby, black);

    // Far-away pixels should remain black
    cv::Vec3b far_away = colored.at<cv::Vec3b>(10, 10);
    EXPECT_EQ(far_away, black);
}

TEST(VisualizerTest, ComparisonWithErrorBuildsDashboardCanvas) {
    PipelineConfig config;
    config.max_disparity = 128.0;
    Visualizer viz(config);

    cv::Mat left(120, 200, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat right(120, 200, CV_8UC3, cv::Scalar(120, 120, 120));
    cv::Mat pred_color(120, 200, CV_8UC3, cv::Scalar(200, 100, 50));
    cv::Mat gt_color(120, 200, CV_8UC3, cv::Scalar(50, 100, 200));
    cv::Mat error_heat(120, 200, CV_8UC3, cv::Scalar(50, 50, 50));

    auto comp = viz.renderComparison(left, right, pred_color, gt_color, error_heat, true);

    // Dashboard should be much larger than a single input image
    EXPECT_GT(comp.rows, 200);
    EXPECT_GT(comp.cols, 600);
    EXPECT_EQ(comp.channels(), 3);

    // Top-left corner should be a dark dashboard background
    cv::Vec3b top_left = comp.at<cv::Vec3b>(0, 0);
    EXPECT_LT(top_left[0], 40);
    EXPECT_LT(top_left[1], 40);
    EXPECT_LT(top_left[2], 40);
}

TEST(VisualizerTest, ErrorHeatmapUsesInfernoAndSparseEnhancement) {
    PipelineConfig config;
    config.max_disparity = 128.0;
    Visualizer viz(config);

    // Sparse valid mask
    cv::Mat pred(80, 120, CV_32F, cv::Scalar(0.0f));
    cv::Mat gt(80, 120, CV_32F, cv::Scalar(0.0f));
    cv::Mat valid_mask = cv::Mat::zeros(80, 120, CV_8UC1);

    pred.at<float>(40, 60) = 50.0f;
    gt.at<float>(40, 60) = 45.0f;
    valid_mask.at<unsigned char>(40, 60) = 255;

    auto heat = viz.renderErrorHeatmap(pred, gt, valid_mask);

    EXPECT_EQ(heat.rows, 80);
    EXPECT_EQ(heat.cols, 120);
    EXPECT_EQ(heat.channels(), 3);

    // Valid pixel should be non-black (INFERNO colormap)
    cv::Vec3b black(0, 0, 0);
    cv::Vec3b error_pixel = heat.at<cv::Vec3b>(40, 60);
    EXPECT_NE(error_pixel, black);

    // Sparse dilation should make nearby pixels visible too (3x3 ellipse)
    cv::Vec3b nearby = heat.at<cv::Vec3b>(40, 61);
    EXPECT_NE(nearby, black);

    // Invalid far-away pixel should remain black
    cv::Vec3b far_pixel = heat.at<cv::Vec3b>(10, 10);
    EXPECT_EQ(far_pixel, black);
}
