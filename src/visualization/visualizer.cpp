#include "visualization/visualizer.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace cms {
namespace {
int toOpenCVColorMap(ColorMap cmap) {
    switch (cmap) {
        case ColorMap::JET: return cv::COLORMAP_JET;
        case ColorMap::MAGMA:
        default: return cv::COLORMAP_INFERNO;
        case ColorMap::TURBO: return cv::COLORMAP_TURBO;
    }
}

cv::Mat toDisplayBGR(const cv::Mat& image) {
    cv::Mat out;
    if (image.empty()) return out;
    cv::Mat norm;
    if (image.depth() == CV_8U) {
        norm = image.clone();
    } else {
        double min_v = 0.0;
        double max_v = 0.0;
        cv::minMaxLoc(image, &min_v, &max_v);
        if (max_v <= min_v) {
            norm = cv::Mat(image.size(), CV_8U, cv::Scalar(0));
        } else {
            image.convertTo(norm, CV_8U, 255.0 / (max_v - min_v), -min_v * 255.0 / (max_v - min_v));
        }
    }
    if (norm.channels() == 1) {
        cv::cvtColor(norm, out, cv::COLOR_GRAY2BGR);
    } else if (norm.channels() == 4) {
        cv::cvtColor(norm, out, cv::COLOR_BGRA2BGR);
    } else {
        out = norm;
    }
    return out;
}

/// Create a vertical color bar image using the specified colormap.
cv::Mat createVerticalColorBar(int height, int width, ColorMap cmap) {
    cv::Mat grad(height, 1, CV_8U);
    for (int y = 0; y < height; ++y) {
        grad.at<uchar>(y, 0) = static_cast<uchar>(255.0 * (1.0 - static_cast<double>(y) / std::max(height - 1, 1)));
    }
    cv::Mat colored;
    cv::applyColorMap(grad, colored, toOpenCVColorMap(cmap));
    cv::resize(colored, colored, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    return colored;
}
}  // namespace

Visualizer::Visualizer(const PipelineConfig& config) : config_(config) {}

cv::Mat Visualizer::renderDisparityColor(const cv::Mat& disparity,
                                         double max_disp,
                                         ColorMap cmap) const {
    // Convert to float for uniform processing
    cv::Mat float_disp;
    if (disparity.depth() == CV_32F) {
        float_disp = disparity.clone();
    } else {
        disparity.convertTo(float_disp, CV_32F);
    }

    // 1. Collect valid positive values and build mask
    std::vector<float> valid_values;
    valid_values.reserve(static_cast<size_t>(float_disp.rows * float_disp.cols) / 4);
    cv::Mat valid_mask = cv::Mat::zeros(float_disp.size(), CV_8UC1);

    for (int y = 0; y < float_disp.rows; ++y) {
        for (int x = 0; x < float_disp.cols; ++x) {
            float v = float_disp.at<float>(y, x);
            if (std::isfinite(v) && v > 0.0f) {
                valid_values.push_back(v);
                valid_mask.at<unsigned char>(y, x) = 255;
            }
        }
    }

    // 2. Robust percentile scaling
    double display_max = max_disp;
    if (!valid_values.empty()) {
        std::sort(valid_values.begin(), valid_values.end());
        size_t idx = static_cast<size_t>(valid_values.size() * 0.95);
        if (idx >= valid_values.size()) idx = valid_values.size() - 1;
        display_max = valid_values[idx];
        display_max = std::min(display_max, max_disp);
        display_max = std::max(display_max, 1.0);
    }

    // 3. Clamp and normalize
    cv::Mat clamped = float_disp.clone();
    for (int y = 0; y < clamped.rows; ++y) {
        for (int x = 0; x < clamped.cols; ++x) {
            float v = clamped.at<float>(y, x);
            if (!std::isfinite(v) || v < 0.0f) {
                clamped.at<float>(y, x) = 0.0f;
            } else if (v > display_max) {
                clamped.at<float>(y, x) = static_cast<float>(display_max);
            }
        }
    }

    cv::Mat normalized;
    clamped.convertTo(normalized, CV_8U, 255.0 / display_max);

    cv::Mat colored;
    cv::applyColorMap(normalized, colored, toOpenCVColorMap(cmap));

    // Black out invalid regions
    colored.setTo(cv::Scalar(0, 0, 0), valid_mask == 0);

    // 4. Sparse data enhancement: dilate valid points for readability
    double valid_ratio = static_cast<double>(valid_values.size()) /
                         static_cast<double>(float_disp.rows * float_disp.cols);
    if (valid_ratio < 0.5 && !valid_values.empty()) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::Mat dilated_mask;
        cv::dilate(valid_mask, dilated_mask, kernel);
        cv::Mat dilated_color;
        cv::dilate(colored, dilated_color, kernel);
        dilated_color.copyTo(colored, dilated_mask);
    }

    return colored;
}

cv::Mat Visualizer::renderErrorHeatmap(const cv::Mat& pred,
                                       const cv::Mat& gt,
                                       const cv::Mat& valid_mask) const {
    cv::Mat error = cv::Mat::zeros(pred.size(), CV_32F);
    for (int y = 0; y < pred.rows; ++y) {
        for (int x = 0; x < pred.cols; ++x) {
            if (valid_mask.at<unsigned char>(y, x) == 0) continue;
            error.at<float>(y, x) = std::abs(pred.at<float>(y, x) - gt.at<float>(y, x));
        }
    }

    // Robust percentile scaling
    std::vector<float> valid_errors;
    for (int y = 0; y < error.rows; ++y) {
        for (int x = 0; x < error.cols; ++x) {
            if (valid_mask.at<unsigned char>(y, x) != 0 && error.at<float>(y, x) > 0.0f) {
                valid_errors.push_back(error.at<float>(y, x));
            }
        }
    }

    double display_max = 10.0;
    if (!valid_errors.empty()) {
        std::sort(valid_errors.begin(), valid_errors.end());
        size_t idx = static_cast<size_t>(valid_errors.size() * 0.95);
        if (idx >= valid_errors.size()) idx = valid_errors.size() - 1;
        display_max = valid_errors[idx];
        display_max = std::max(display_max, 0.1);
    }

    cv::Mat clamped;
    cv::min(error, display_max, clamped);
    cv::max(clamped, 0.0, clamped);

    cv::Mat normalized;
    clamped.convertTo(normalized, CV_8U, 255.0 / display_max);

    cv::Mat heatmap;
    cv::applyColorMap(normalized, heatmap, cv::COLORMAP_INFERNO);
    heatmap.setTo(cv::Scalar(0, 0, 0), valid_mask == 0);

    // Sparse enhancement
    double valid_ratio = static_cast<double>(valid_errors.size()) /
                         static_cast<double>(error.rows * error.cols);
    if (valid_ratio < 0.5 && !valid_errors.empty()) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::Mat dilated_mask;
        cv::dilate(valid_mask, dilated_mask, kernel);
        cv::Mat dilated;
        cv::dilate(heatmap, dilated, kernel);
        dilated.copyTo(heatmap, dilated_mask);
    }

    return heatmap;
}

cv::Mat Visualizer::createOverlay(const cv::Mat& left_rgb, const cv::Mat& pred_color) const {
    cv::Mat left = toDisplayBGR(left_rgb);
    cv::Mat pred = toDisplayBGR(pred_color);
    cv::resize(pred, pred, left.size());

    // Detect non-black pixels in pred as valid disparity region
    cv::Mat gray;
    cv::cvtColor(pred, gray, cv::COLOR_BGR2GRAY);
    cv::Mat valid_region = gray > 10;

    cv::Mat overlay;
    cv::addWeighted(left, 0.5, pred, 0.5, 0, overlay);
    overlay.copyTo(overlay, valid_region);
    left.copyTo(overlay, ~valid_region);

    return overlay;
}

cv::Mat Visualizer::buildDashboardPanel(const cv::Mat& content,
                                        const std::string& title,
                                        const std::string& subtitle,
                                        int target_w,
                                        int target_h,
                                        bool add_colorbar,
                                        const std::string& colorbar_label,
                                        double colorbar_min,
                                        double colorbar_max,
                                        ColorMap cmap) const {
    const int title_h = 32;
    const int subtitle_h = 20;
    const int padding = 8;
    const int colorbar_w = add_colorbar ? 56 : 0;

    int content_w = target_w - 2 * padding - colorbar_w;
    int content_h = target_h - title_h - subtitle_h - 2 * padding;
    if (content_w < 10) content_w = 10;
    if (content_h < 10) content_h = 10;

    cv::Mat panel(target_h, target_w, CV_8UC3, cv::Scalar(35, 35, 35));

    // Title bar
    cv::rectangle(panel, cv::Rect(0, 0, target_w, title_h), cv::Scalar(28, 28, 28), cv::FILLED);
    cv::putText(panel, title, cv::Point(padding, 23), cv::FONT_HERSHEY_SIMPLEX, 0.60,
                cv::Scalar(230, 230, 230), 1, cv::LINE_AA);

    // Content area: preserve aspect ratio
    cv::Mat resized;
    if (!content.empty()) {
        double scale = std::min(static_cast<double>(content_w) / content.cols,
                                static_cast<double>(content_h) / content.rows);
        int new_w = static_cast<int>(content.cols * scale);
        int new_h = static_cast<int>(content.rows * scale);
        cv::resize(content, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);

        int offset_x = padding + (content_w - new_w) / 2;
        int offset_y = title_h + padding + (content_h - new_h) / 2;
        cv::Rect roi(offset_x, offset_y, new_w, new_h);
        resized.copyTo(panel(roi));
    }

    // Colorbar
    if (add_colorbar) {
        int cb_x = target_w - padding - colorbar_w + 6;
        int cb_y = title_h + padding;
        int cb_h = content_h;
        int cb_bar_w = 14;

        cv::Mat bar = createVerticalColorBar(cb_h, cb_bar_w, cmap);
        cv::Rect bar_roi(cb_x, cb_y, cb_bar_w, cb_h);
        bar.copyTo(panel(bar_roi));

        // Border around colorbar
        cv::rectangle(panel, bar_roi, cv::Scalar(80, 80, 80), 1);

        // Labels
        cv::putText(panel, cv::format("%.1f%s", colorbar_max, colorbar_label.c_str()),
                    cv::Point(cb_x + cb_bar_w + 5, cb_y + 11),
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(160, 160, 160), 1, cv::LINE_AA);
        cv::putText(panel, cv::format("%.1f", colorbar_min),
                    cv::Point(cb_x + cb_bar_w + 5, cb_y + cb_h),
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(160, 160, 160), 1, cv::LINE_AA);
    }

    // Subtitle
    cv::putText(panel, subtitle, cv::Point(padding, target_h - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(140, 140, 140), 1, cv::LINE_AA);

    return panel;
}

cv::Mat Visualizer::renderComparison(const cv::Mat& left_rgb,
                                     const cv::Mat& right_ir,
                                     const cv::Mat& pred_color,
                                     const cv::Mat& gt_color,
                                     const cv::Mat& error_heatmap,
                                     bool gt_available) const {
    cv::Mat left = toDisplayBGR(left_rgb);
    cv::Mat right = toDisplayBGR(right_ir);
    cv::Mat pred = toDisplayBGR(pred_color);
    cv::Mat gt = gt_available ? toDisplayBGR(gt_color)
                              : cv::Mat(pred.size(), CV_8UC3, cv::Scalar(60, 60, 60));

    cv::resize(right, right, left.size());
    cv::resize(pred, pred, left.size());
    cv::resize(gt, gt, left.size());

    cv::Mat overlay = createOverlay(left_rgb, pred_color);
    cv::resize(overlay, overlay, left.size());

    const int panel_w = 420;
    const int panel_h = 310;
    const int gap = 10;

    cv::Mat dashboard(2 * panel_h + gap, 3 * panel_w + 2 * gap, CV_8UC3, cv::Scalar(20, 20, 20));

    auto placePanel = [&](const cv::Mat& panel, int row, int col) {
        int x = col * (panel_w + gap);
        int y = row * (panel_h + gap);
        if (x + panel.cols <= dashboard.cols && y + panel.rows <= dashboard.rows) {
            panel.copyTo(dashboard(cv::Rect(x, y, panel.cols, panel.rows)));
        }
    };

    placePanel(buildDashboardPanel(left, "Left RGB", "Reference image",
                                   panel_w, panel_h, false), 0, 0);
    placePanel(buildDashboardPanel(right, "Right IR", "Cross-modal stereo pair",
                                   panel_w, panel_h, false), 0, 1);
    placePanel(buildDashboardPanel(overlay, "Prediction overlay",
                                   "Disparity color blended over left RGB",
                                   panel_w, panel_h, false), 0, 2);
    placePanel(buildDashboardPanel(pred, "Predicted disparity",
                                   "Robust percentile stretch",
                                   panel_w, panel_h, true, "", 0.0, config_.max_disparity),
               1, 0);
    placePanel(buildDashboardPanel(gt, "Ground truth",
                                   "Sparse valid LiDAR points enlarged for readability",
                                   panel_w, panel_h, true, "", 0.0, config_.max_disparity),
               1, 1);

    if (!error_heatmap.empty()) {
        cv::Mat err = toDisplayBGR(error_heatmap);
        cv::resize(err, err, left.size());
        placePanel(buildDashboardPanel(err, "Absolute error", "Valid GT pixels only",
                                       panel_w, panel_h, true, "", 0.0, 0.0, ColorMap::MAGMA),
                   1, 2);
    } else {
        placePanel(buildDashboardPanel(cv::Mat(left.size(), CV_8UC3, cv::Scalar(25, 25, 25)),
                                       "Absolute error", "N/A", panel_w, panel_h,
                                       false, "", 0.0, 0.0, ColorMap::TURBO),
                   1, 2);
    }

    return dashboard;
}

cv::Mat Visualizer::renderComparison(const cv::Mat& left_rgb,
                                     const cv::Mat& right_ir,
                                     const cv::Mat& pred_color,
                                     const cv::Mat& gt_color,
                                     bool gt_available) const {
    cv::Mat empty_err;
    return renderComparison(left_rgb, right_ir, pred_color, gt_color, empty_err, gt_available);
}

void Visualizer::save(const cv::Mat& image,
                      const std::string& sample_id,
                      const std::string& suffix,
                      const std::filesystem::path& output_dir) const {
    std::filesystem::create_directories(output_dir);
    const auto out = output_dir / (sample_id + "_" + suffix + ".png");
    cv::imwrite(out.string(), image);
}

}  // namespace cms
