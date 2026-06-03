#include "aggregation/cost_aggregator.h"

#include <algorithm>
#include <array>
#include <limits>
#include <tuple>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace cms {
namespace {
std::vector<std::pair<int, int>> makeDirections(int num_directions) {
    std::vector<std::pair<int, int>> dirs = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    };
    if (num_directions >= 8) {
        dirs.insert(dirs.end(), {{1, 1}, {-1, -1}, {1, -1}, {-1, 1}});
    }
    if (num_directions >= 16) {
        dirs.insert(dirs.end(), {
            {2, 1}, {-2, -1}, {2, -1}, {-2, 1},
            {1, 2}, {-1, -2}, {1, -2}, {-1, 2},
        });
    }
    return dirs;
}

float refValueAt(const cv::Mat& ref, int y, int x) {
    if (ref.empty()) return 0.0f;
    if (ref.type() == CV_32F) return ref.at<float>(y, x);
    cv::Mat tmp;
    ref.convertTo(tmp, CV_32F);
    return tmp.at<float>(y, x);
}
}  // namespace

SGMAggregator::SGMAggregator(int p1, int p2, int num_directions)
    : p1_(p1), p2_(p2), num_directions_(num_directions) {}

Result<CostVolume> SGMAggregator::aggregate(
    const CostVolume& cost_volume,
    const cv::Mat& reference_image) const {
    CostVolume aggregated;
    aggregated.height = cost_volume.height;
    aggregated.width = cost_volume.width;
    aggregated.max_disparity = cost_volume.max_disparity;
    aggregated.data = cv::Mat(cost_volume.data.size(), CV_32F, cv::Scalar(0));

    for (const auto& [dx, dy] : makeDirections(num_directions_)) {
        aggregatePath(cost_volume, aggregated, reference_image, dx, dy);
    }
    return Result<CostVolume>::success(aggregated);
}

void SGMAggregator::aggregatePath(const CostVolume& cost,
                                  CostVolume& aggregated,
                                  const cv::Mat& ref_img,
                                  int dx, int dy) const {
    std::vector<std::pair<int, int>> starts;
    for (int y = 0; y < cost.height; ++y) {
        for (int x = 0; x < cost.width; ++x) {
            const int px = x - dx;
            const int py = y - dy;
            if (px < 0 || px >= cost.width || py < 0 || py >= cost.height) {
                starts.emplace_back(x, y);
            }
        }
    }

    for (const auto& [sx, sy] : starts) {
        std::vector<float> prev(cost.max_disparity, 0.0f);
        bool first = true;
        int x = sx;
        int y = sy;
        while (x >= 0 && x < cost.width && y >= 0 && y < cost.height) {
            std::vector<float> curr(cost.max_disparity, 0.0f);
            if (first) {
                for (int d = 0; d < cost.max_disparity; ++d) {
                    curr[d] = cost.at(y, x, d);
                    aggregated.at(y, x, d) += curr[d];
                }
                first = false;
            } else {
                const int px = x - dx;
                const int py = y - dy;
                const float prev_min = *std::min_element(prev.begin(), prev.end());
                const float grad = std::abs(refValueAt(ref_img, y, x) - refValueAt(ref_img, py, px));
                const float adaptive_p2 = static_cast<float>(p2_) / (1.0f + grad * 8.0f);
                for (int d = 0; d < cost.max_disparity; ++d) {
                    float v0 = prev[d];
                    float v1 = d > 0 ? prev[d - 1] + static_cast<float>(p1_) : std::numeric_limits<float>::max();
                    float v2 = d + 1 < cost.max_disparity ? prev[d + 1] + static_cast<float>(p1_) : std::numeric_limits<float>::max();
                    float v3 = prev_min + adaptive_p2;
                    curr[d] = cost.at(y, x, d) + std::min({v0, v1, v2, v3}) - prev_min;
                    aggregated.at(y, x, d) += curr[d];
                }
            }
            prev.swap(curr);
            x += dx;
            y += dy;
        }
    }
}

GuidedFilterAggregator::GuidedFilterAggregator(int radius, double eps)
    : radius_(radius), eps_(eps) {
    (void)eps_;
}

Result<CostVolume> GuidedFilterAggregator::aggregate(
    const CostVolume& cost_volume,
    const cv::Mat& /*reference_image*/) const {
    CostVolume aggregated;
    aggregated.height = cost_volume.height;
    aggregated.width = cost_volume.width;
    aggregated.max_disparity = cost_volume.max_disparity;
    aggregated.data = cv::Mat(cost_volume.data.size(), CV_32F, cv::Scalar(0));

    for (int d = 0; d < cost_volume.max_disparity; ++d) {
        cv::Mat slice(cost_volume.height, cost_volume.width, CV_32F);
        for (int y = 0; y < cost_volume.height; ++y) {
            for (int x = 0; x < cost_volume.width; ++x) {
                slice.at<float>(y, x) = cost_volume.at(y, x, d);
            }
        }
        cv::Mat filtered;
        cv::blur(slice, filtered, cv::Size(radius_, radius_));
        for (int y = 0; y < cost_volume.height; ++y) {
            for (int x = 0; x < cost_volume.width; ++x) {
                aggregated.at(y, x, d) = filtered.at<float>(y, x);
            }
        }
    }
    return Result<CostVolume>::success(aggregated);
}

std::unique_ptr<ICostAggregator> createAggregator(const PipelineConfig& config) {
    switch (config.aggregation_method) {
        case AggregationMethod::GUIDED_FILTER:
            return std::make_unique<GuidedFilterAggregator>();
        case AggregationMethod::SGM:
        default:
            return std::make_unique<SGMAggregator>(config.sgm_p1, config.sgm_p2, config.sgm_num_directions);
    }
}

}  // namespace cms
