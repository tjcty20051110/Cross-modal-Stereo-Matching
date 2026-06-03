#pragma once

#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace cms {

struct EvalMetrics {
    double mse = std::numeric_limits<double>::quiet_NaN();
    double epe = std::numeric_limits<double>::quiet_NaN();
    double d1_all = std::numeric_limits<double>::quiet_NaN();
    int valid_pixel_count = 0;
    bool is_valid = false;
    std::string sample_id;
};

struct SplitMetrics {
    EvalMetrics average;
    std::vector<EvalMetrics> per_sample;
    int total_samples = 0;
    int valid_samples = 0;
};

class Evaluator {
public:
    static EvalMetrics evaluate(
        const cv::Mat& pred,
        const cv::Mat& gt,
        const cv::Mat& valid_mask,
        const std::string& sample_id = "");

    static SplitMetrics summarize(const std::vector<EvalMetrics>& metrics);
    static void writeCSV(const SplitMetrics& metrics,
                         const std::filesystem::path& output_path);
};

}  // namespace cms
