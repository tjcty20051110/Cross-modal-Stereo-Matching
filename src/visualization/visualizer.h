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

    /// Dashboard-style comparison including error heatmap (new overload)
    cv::Mat renderComparison(const cv::Mat& left_rgb,
                             const cv::Mat& right_ir,
                             const cv::Mat& pred_color,
                             const cv::Mat& gt_color,
                             const cv::Mat& error_heatmap,
                             bool gt_available = true) const;

    /// Legacy overload without error heatmap
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

    cv::Mat createOverlay(const cv::Mat& left_rgb, const cv::Mat& pred_color) const;
    cv::Mat buildDashboardPanel(const cv::Mat& content,
                                const std::string& title,
                                const std::string& subtitle,
                                int target_w,
                                int target_h,
                                bool add_colorbar = false,
                                const std::string& colorbar_label = "",
                                double colorbar_min = 0.0,
                                double colorbar_max = 0.0,
                                ColorMap cmap = ColorMap::TURBO) const;
};

}  // namespace cms
