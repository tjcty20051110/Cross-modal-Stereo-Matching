#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "data/data_loader.h"

namespace {

std::filesystem::path makeTempDir(const std::string& name) {
    auto dir = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

void writeCalibration(const std::filesystem::path& path) {
    std::ofstream out(path);
    out << "focal_length: 320.0\n";
    out << "baseline: 0.2\n";
}

}  // namespace

TEST(DataLoaderTest, LoadsManifestAndSample) {
    auto dir = makeTempDir("cms_test_data_loader_ok");
    auto left = dir / "left.png";
    auto right = dir / "right_nir.png";
    auto gt = dir / "gt.png";
    auto calib = dir / "calib.yaml";
    auto manifest = dir / "manifest.csv";

    cv::imwrite(left.string(), cv::Mat(4, 5, CV_8UC3, cv::Scalar(10, 20, 30)));
    cv::imwrite(right.string(), cv::Mat(4, 5, CV_8UC1, cv::Scalar(50)));
    cv::imwrite(gt.string(), cv::Mat(4, 5, CV_16UC1, cv::Scalar(4)));
    writeCalibration(calib);

    std::ofstream out(manifest);
    out << "left_path,right_path,gt_disparity_path,calib_path,split\n";
    out << "left.png,right_nir.png,gt.png,calib.yaml,train\n";
    out.close();

    cms::DataLoader loader;
    auto count = loader.loadManifest(manifest);
    ASSERT_TRUE(count.has_value());
    EXPECT_EQ(count.value(), 1u);

    auto sample = loader.getSample(0);
    ASSERT_TRUE(sample.has_value());
    EXPECT_EQ(sample.value().left_image.rows, 4);
    EXPECT_EQ(sample.value().left_image.cols, 5);
    EXPECT_EQ(sample.value().right_image.size(), sample.value().left_image.size());
    EXPECT_EQ(sample.value().camera.focal_length, 320.0);
    EXPECT_EQ(sample.value().camera.baseline, 0.2);
    EXPECT_FALSE(sample.value().gt_disparity.empty());
    EXPECT_GT(cv::countNonZero(sample.value().valid_mask), 0);

    std::filesystem::remove_all(dir);
}

TEST(DataLoaderTest, MissingFilePathAppearsInError) {
    auto dir = makeTempDir("cms_test_data_loader_missing");
    auto manifest = dir / "manifest.csv";
    std::ofstream out(manifest);
    out << "left_path,right_path,gt_disparity_path,calib_path,split\n";
    out << "missing_left.png,missing_right.png,,calib.yaml,val\n";
    out.close();

    cms::DataLoader loader;
    auto count = loader.loadManifest(manifest);
    ASSERT_TRUE(count.has_value());
    auto sample = loader.getSample(0);
    EXPECT_FALSE(sample.has_value());
    EXPECT_NE(sample.error_msg().find("missing_left.png"), std::string::npos);

    std::filesystem::remove_all(dir);
}

TEST(DataLoaderTest, ZeroGroundTruthIsTreatedAsMissing) {
    auto dir = makeTempDir("cms_test_data_loader_zero_gt");
    auto left = dir / "left.png";
    auto right = dir / "right_ir.png";
    auto gt = dir / "gt_zero.png";
    auto calib = dir / "calib.yaml";
    auto manifest = dir / "manifest.csv";

    cv::imwrite(left.string(), cv::Mat(3, 3, CV_8UC3, cv::Scalar(10, 10, 10)));
    cv::imwrite(right.string(), cv::Mat(3, 3, CV_8UC1, cv::Scalar(10)));
    cv::imwrite(gt.string(), cv::Mat::zeros(3, 3, CV_16UC1));
    writeCalibration(calib);

    std::ofstream out(manifest);
    out << "left_path,right_path,gt_disparity_path,calib_path,split\n";
    out << "left.png,right_ir.png,gt_zero.png,calib.yaml,test\n";
    out.close();

    cms::DataLoader loader;
    ASSERT_TRUE(loader.loadManifest(manifest).has_value());
    auto sample = loader.getSample(0);
    ASSERT_TRUE(sample.has_value());
    EXPECT_TRUE(sample.value().gt_disparity.empty());
    EXPECT_EQ(cv::countNonZero(sample.value().valid_mask), 0);

    std::filesystem::remove_all(dir);
}
