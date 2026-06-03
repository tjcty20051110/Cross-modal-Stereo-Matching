#include "cost/cost_volume_builder.h"

#include <cmath>
#include <limits>

namespace cms {

float& CostVolume::at(int y, int x, int d) {
    return data.at<float>(y * max_disparity + d, x);
}

float CostVolume::at(int y, int x, int d) const {
    return data.at<float>(y * max_disparity + d, x);
}

CostVolumeBuilder::CostVolumeBuilder(const PipelineConfig& config) : config_(config) {}

Result<CostVolume> CostVolumeBuilder::build(
    const FeatureTensor& left_feat,
    const FeatureTensor& right_feat,
    int max_disparity) const {
    if (left_feat.data.empty() || right_feat.data.empty()) {
        return Result<CostVolume>::error("Empty feature tensor provided to CostVolumeBuilder");
    }
    if (left_feat.data.size() != right_feat.data.size() || left_feat.channels != right_feat.channels) {
        return Result<CostVolume>::error("Feature tensor shape mismatch");
    }
    if (max_disparity < 1 || max_disparity > left_feat.data.cols) {
        return Result<CostVolume>::error("Invalid max_disparity: " + std::to_string(max_disparity));
    }

    CostVolume volume;
    volume.height = left_feat.data.rows;
    volume.width = left_feat.data.cols;
    volume.max_disparity = max_disparity;
    volume.data = cv::Mat(volume.height * volume.max_disparity, volume.width, CV_32F, cv::Scalar(0));

    const float invalid_cost = static_cast<float>(std::max(1, left_feat.channels));
    for (int y = 0; y < volume.height; ++y) {
        for (int x = 0; x < volume.width; ++x) {
            for (int d = 0; d < max_disparity; ++d) {
                const int xr = x - d;
                volume.at(y, x, d) = xr >= 0 ? computeCost(left_feat.data, right_feat.data, y, x, xr) : invalid_cost;
            }
        }
    }

    return Result<CostVolume>::success(volume);
}

Result<std::pair<CostVolume, CostVolume>> CostVolumeBuilder::buildSymmetric(
    const FeatureTensor& left_feat,
    const FeatureTensor& right_feat,
    int max_disparity) const {
    auto left = build(left_feat, right_feat, max_disparity);
    if (!left) {
        return Result<std::pair<CostVolume, CostVolume>>::error(left.error_msg());
    }

    CostVolume right;
    right.height = right_feat.data.rows;
    right.width = right_feat.data.cols;
    right.max_disparity = max_disparity;
    right.data = cv::Mat(right.height * right.max_disparity, right.width, CV_32F, cv::Scalar(0));

    const float invalid_cost = static_cast<float>(std::max(1, right_feat.channels));
    for (int y = 0; y < right.height; ++y) {
        for (int x = 0; x < right.width; ++x) {
            for (int d = 0; d < max_disparity; ++d) {
                const int xl = x + d;
                right.at(y, x, d) = xl < right.width ? computeCost(right_feat.data, left_feat.data, y, x, xl) : invalid_cost;
            }
        }
    }

    return Result<std::pair<CostVolume, CostVolume>>::success({left.value(), right});
}

float CostVolumeBuilder::computeCensusHamming(const cv::Mat& left, const cv::Mat& right,
                                              int y, int x_left, int x_right) const {
    const float* l = left.ptr<float>(y, x_left);
    const float* r = right.ptr<float>(y, x_right);
    const int channels = left.channels();
    int distance = 0;
    for (int c = 0; c < channels; ++c) {
        const bool lb = l[c] > 0.5f;
        const bool rb = r[c] > 0.5f;
        distance += (lb != rb) ? 1 : 0;
    }
    return static_cast<float>(distance);
}

float CostVolumeBuilder::computeNCC(const cv::Mat& left, const cv::Mat& right,
                                    int y, int x_left, int x_right) const {
    const float* l = left.ptr<float>(y, x_left);
    const float* r = right.ptr<float>(y, x_right);
    const int channels = left.channels();
    float mean_l = 0.0f;
    float mean_r = 0.0f;
    for (int c = 0; c < channels; ++c) {
        mean_l += l[c];
        mean_r += r[c];
    }
    mean_l /= static_cast<float>(channels);
    mean_r /= static_cast<float>(channels);
    float numerator = 0.0f;
    float denom_l = 0.0f;
    float denom_r = 0.0f;
    for (int c = 0; c < channels; ++c) {
        const float dl = l[c] - mean_l;
        const float dr = r[c] - mean_r;
        numerator += dl * dr;
        denom_l += dl * dl;
        denom_r += dr * dr;
    }
    const float corr = numerator / std::sqrt((denom_l + static_cast<float>(EPSILON)) * (denom_r + static_cast<float>(EPSILON)));
    return 1.0f - corr;
}

float CostVolumeBuilder::computeCosineSimilarity(const cv::Mat& left, const cv::Mat& right,
                                                 int y, int x_left, int x_right) const {
    const float* l = left.ptr<float>(y, x_left);
    const float* r = right.ptr<float>(y, x_right);
    const int channels = left.channels();
    float dot = 0.0f;
    float norm_l = 0.0f;
    float norm_r = 0.0f;
    for (int c = 0; c < channels; ++c) {
        dot += l[c] * r[c];
        norm_l += l[c] * l[c];
        norm_r += r[c] * r[c];
    }
    const float cosine = dot / std::sqrt((norm_l + static_cast<float>(EPSILON)) * (norm_r + static_cast<float>(EPSILON)));
    return 1.0f - cosine;
}

float CostVolumeBuilder::computeCost(const cv::Mat& left, const cv::Mat& right,
                                     int y, int x_left, int x_right) const {
    switch (config_.cost_metric) {
        case CostMetric::NCC:
            return computeNCC(left, right, y, x_left, x_right);
        case CostMetric::COSINE_SIMILARITY:
            return computeCosineSimilarity(left, right, y, x_left, x_right);
        case CostMetric::CENSUS_HAMMING:
        default:
            return computeCensusHamming(left, right, y, x_left, x_right);
    }
}

}  // namespace cms
