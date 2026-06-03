#pragma once

#include <filesystem>

#include <opencv2/core.hpp>

#include "config/config_manager.h"

namespace cms {

enum class ColorMap { JET, TURBO, MAGMA };

class Visualizer {
public:
    explicit Visualizer(const PipelineConfig& config);

    cv::Mat renderDisparityColor(const cv::Mat& disparity,
                                 double max_disp,
                                 ColorMap cmap = ColorMap::TURBO) const;
    cv::Mat renderErrorHeatmap(const cv::Mat& pred,
                               const cv::Mat& gt,
                               const cv::Mat& valid_mask) const;
    cv::Mat renderComparison(const cv::Mat& left_rgb,
                             const cv::Mat& right_ir,
                             const cv::Mat& pred_color,
                             const cv::Mat& gt_color,
                             bool gt_available = true) const;
    void save(const cv::Mat& image,
              const std::string& sample_id,
              const std::string& suffix,
              const std::filesystem::path& output_dir) const;

private:
    PipelineConfig config_;
};

}  // namespace cms
