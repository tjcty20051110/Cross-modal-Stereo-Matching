// config_manager.h — Configuration management for the CMS pipeline
#pragma once

#include <string>
#include <filesystem>
#include "common/types.h"

namespace cms {

// ---------------------------------------------------------------------------
// Core enumerations
// ---------------------------------------------------------------------------

/// Feature extraction method selection
enum class FeatureMethod {
    CENSUS,                 ///< Census transform (default, modality-invariant)
    GRADIENT_ORIENTATION,   ///< Gradient orientation descriptor
    DL_FEATURE              ///< Deep learning feature extraction
};

/// Matching cost metric
enum class CostMetric {
    NCC,                    ///< Normalized Cross-Correlation
    CENSUS_HAMMING,         ///< Hamming distance on Census descriptors
    COSINE_SIMILARITY       ///< Cosine similarity on feature vectors
};

/// Cost aggregation method
enum class AggregationMethod {
    SGM,                    ///< Semi-Global Matching
    GUIDED_FILTER           ///< Guided filter aggregation
};

/// Deep learning inference backend
enum class InferenceBackend {
    ONNX_RUNTIME,           ///< ONNX Runtime
    LIBTORCH,               ///< LibTorch (PyTorch C++ frontend)
    OPENCV_DNN,             ///< OpenCV DNN module
    TENSORRT                ///< NVIDIA TensorRT
};

// ---------------------------------------------------------------------------
// Pipeline configuration structure
// ---------------------------------------------------------------------------

/// Complete pipeline configuration with sensible defaults.
/// All optional parameters have default values; mandatory parameters
/// (e.g., manifest_path for run mode) are validated at runtime.
struct PipelineConfig {
    // --- Core matching parameters ---
    int max_disparity = 128;
    FeatureMethod feature_method = FeatureMethod::CENSUS;
    CostMetric cost_metric = CostMetric::CENSUS_HAMMING;
    AggregationMethod aggregation_method = AggregationMethod::SGM;

    // --- SGM parameters ---
    int sgm_num_directions = 8;     ///< Number of scan directions (4, 8, or 16)
    int sgm_p1 = 10;               ///< Small disparity change penalty
    int sgm_p2 = 120;              ///< Large disparity change penalty

    // --- Post-processing ---
    bool enable_lr_check = true;    ///< Enable left-right consistency check
    bool enable_hole_filling = true;///< Enable hole filling for invalid pixels
    double lr_threshold = 1.0;      ///< LR check threshold (pixels)

    // --- Depth conversion ---
    double depth_max_meter = 100.0; ///< Maximum depth clamp (meters)

    // --- Preprocessing ---
    bool enable_clahe = true;       ///< Enable CLAHE contrast enhancement
    int clahe_clip_limit = 4;       ///< CLAHE clip limit
    int clahe_tile_size = 8;        ///< CLAHE tile grid size
    bool enable_gradient_alignment = false; ///< Enable gradient domain alignment

    // --- Deep learning path ---
    bool use_dl_features = false;           ///< Use DL feature extraction
    bool dl_fallback_to_handcraft = true;   ///< Fallback to handcraft on DL failure
    InferenceBackend inference_backend = InferenceBackend::ONNX_RUNTIME;
    std::filesystem::path model_weights_path; ///< Path to model weights file
    bool use_gpu = false;                   ///< Prefer GPU inference if available

    // --- I/O paths ---
    std::filesystem::path output_dir = "./output";  ///< Output directory
    std::filesystem::path manifest_path;            ///< Dataset manifest file path
};

// ---------------------------------------------------------------------------
// ConfigManager class
// ---------------------------------------------------------------------------

/// Manages loading, validation, and logging of pipeline configuration.
class ConfigManager {
public:
    /// Load configuration from a YAML file.
    /// Missing optional parameters are filled with PipelineConfig defaults.
    /// @param config_path Path to the YAML configuration file
    /// @return PipelineConfig on success, error message on failure
    static Result<PipelineConfig> load(const std::filesystem::path& config_path);

    /// Validate parameter ranges and constraints.
    /// Checks: max_disparity > 0, P2 > P1 >= 0, depth_max_meter > 0, etc.
    /// @param config The configuration to validate
    /// @return Success or error message containing the violating field name
    static Result<void> validate(const PipelineConfig& config);

    /// Log the effective configuration summary to stdout.
    /// @param config The configuration to summarize
    static void logSummary(const PipelineConfig& config);
};

} // namespace cms
