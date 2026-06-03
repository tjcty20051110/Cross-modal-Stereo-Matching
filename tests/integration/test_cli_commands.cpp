#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "cli/cli.h"

namespace {
std::filesystem::path makeCliFixture(const std::string& name) {
    auto dir = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

void writeFixtureData(const std::filesystem::path& dir) {
    cv::Mat left(24, 24, CV_8UC3, cv::Scalar(20, 40, 60));
    cv::Mat right(24, 24, CV_8UC1, cv::Scalar(40));
    cv::Mat gt(24, 24, CV_16UC1, cv::Scalar(1));
    cv::imwrite((dir / "left.png").string(), left);
    cv::imwrite((dir / "right_nir.png").string(), right);
    cv::imwrite((dir / "gt.png").string(), gt);

    std::ofstream calib(dir / "calib.yaml");
    calib << "focal_length: 320.0\n";
    calib << "baseline: 0.2\n";
    calib.close();

    std::ofstream manifest(dir / "manifest.csv");
    manifest << "left_path,right_path,gt_disparity_path,calib_path,split\n";
    manifest << "left.png,right_nir.png,gt.png,calib.yaml,val\n";
    manifest.close();

    std::ofstream config(dir / "config.yaml");
    config << R"(max_disparity: 8
feature_method: CENSUS
cost_metric: CENSUS_HAMMING
aggregation_method: SGM
sgm:
  num_directions: 4
  p1: 5
  p2: 20
post_processing:
  enable_lr_check: true
  enable_hole_filling: true
  lr_threshold: 1.0
depth:
  depth_max_meter: 100.0
preprocessing:
  enable_clahe: false
  clahe_clip_limit: 4
  clahe_tile_size: 8
  enable_gradient_alignment: false
dl:
  use_dl_features: false
  fallback_to_handcraft: true
  inference_backend: OPENCV_DNN
  model_weights_path: ''
  use_gpu: false
)";
    config << "io:\n";
    config << "  manifest_path: '" << (dir / "manifest.csv").string() << "'\n";
    config << "  output_dir: '" << (dir / "output").string() << "'\n";
    config.close();
}

int runCli(std::vector<std::string> argv) {
    std::vector<char*> raw;
    raw.reserve(argv.size());
    for (auto& arg : argv) raw.push_back(arg.data());
    cms::CLI cli;
    return cli.run(static_cast<int>(raw.size()), raw.data());
}
}  // namespace

TEST(CLIIntegrationTest, UnknownCommandReturnsFailure) {
    EXPECT_EQ(runCli({"cms", "unknown"}), 1);
}

TEST(CLIIntegrationTest, RunCommandProducesOutputs) {
    auto dir = makeCliFixture("cms_cli_run");
    writeFixtureData(dir);

    EXPECT_EQ(runCli({"cms", "run", "--config", (dir / "config.yaml").string()}), 0);
    EXPECT_TRUE(std::filesystem::exists(dir / "output" / "left_0_disp.png"));
    EXPECT_TRUE(std::filesystem::exists(dir / "output" / "left_0_mask.png"));

    std::filesystem::remove_all(dir);
}

TEST(CLIIntegrationTest, AnalyzeCommandProducesReportArtifacts) {
    auto dir = makeCliFixture("cms_cli_analyze");
    writeFixtureData(dir);

    EXPECT_EQ(runCli({"cms", "analyze", "--config", (dir / "config.yaml").string()}), 0);
    EXPECT_TRUE(std::filesystem::exists(dir / "output" / "analysis" / "modality_stats.csv"));
    EXPECT_TRUE(std::filesystem::exists(dir / "output" / "analysis" / "cost_metric_comparison.csv"));
    EXPECT_TRUE(std::filesystem::exists(dir / "output" / "analysis" / "modality_analysis_report.md"));

    std::filesystem::remove_all(dir);
}
