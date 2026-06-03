#include "evaluation/evaluator.h"

#include <fstream>

namespace cms {

EvalMetrics Evaluator::evaluate(
    const cv::Mat& pred,
    const cv::Mat& gt,
    const cv::Mat& valid_mask,
    const std::string& sample_id) {
    EvalMetrics metrics;
    metrics.sample_id = sample_id;
    if (pred.empty() || gt.empty() || valid_mask.empty()) {
        return metrics;
    }

    double mse_sum = 0.0;
    double epe_sum = 0.0;
    int d1_count = 0;
    int valid_count = 0;

    for (int y = 0; y < pred.rows; ++y) {
        for (int x = 0; x < pred.cols; ++x) {
            if (valid_mask.at<unsigned char>(y, x) == 0) continue;
            const double p = pred.at<float>(y, x);
            const double g = gt.at<float>(y, x);
            const double diff = std::abs(p - g);
            mse_sum += diff * diff;
            epe_sum += diff;
            d1_count += diff > std::max(3.0, 0.05 * g) ? 1 : 0;
            ++valid_count;
        }
    }

    metrics.valid_pixel_count = valid_count;
    metrics.is_valid = valid_count > 0;
    if (valid_count > 0) {
        metrics.mse = mse_sum / valid_count;
        metrics.epe = epe_sum / valid_count;
        metrics.d1_all = 100.0 * static_cast<double>(d1_count) / valid_count;
    }
    return metrics;
}

SplitMetrics Evaluator::summarize(const std::vector<EvalMetrics>& metrics) {
    SplitMetrics split;
    split.per_sample = metrics;
    split.total_samples = static_cast<int>(metrics.size());

    double mse_sum = 0.0;
    double epe_sum = 0.0;
    double d1_sum = 0.0;
    for (const auto& metric : metrics) {
        if (!metric.is_valid) continue;
        split.valid_samples += 1;
        mse_sum += metric.mse;
        epe_sum += metric.epe;
        d1_sum += metric.d1_all;
    }
    if (split.valid_samples > 0) {
        split.average.mse = mse_sum / split.valid_samples;
        split.average.epe = epe_sum / split.valid_samples;
        split.average.d1_all = d1_sum / split.valid_samples;
        split.average.is_valid = true;
    }
    split.average.valid_pixel_count = split.valid_samples;
    split.average.sample_id = "average";
    return split;
}

void Evaluator::writeCSV(const SplitMetrics& metrics,
                         const std::filesystem::path& output_path) {
    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream out(output_path);
    out << "sample_id,mse,epe,d1_all,valid_pixel_count,is_valid\n";
    for (const auto& metric : metrics.per_sample) {
        out << metric.sample_id << ','
            << metric.mse << ','
            << metric.epe << ','
            << metric.d1_all << ','
            << metric.valid_pixel_count << ','
            << metric.is_valid << '\n';
    }
    out << metrics.average.sample_id << ','
        << metrics.average.mse << ','
        << metrics.average.epe << ','
        << metrics.average.d1_all << ','
        << metrics.average.valid_pixel_count << ','
        << metrics.average.is_valid << '\n';
}

}  // namespace cms
