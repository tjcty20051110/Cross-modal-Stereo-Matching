// cfm_inference.h — Inference driver for the CFM pipeline
#pragma once

#ifdef CMS_HAS_LIBTORCH

#include <filesystem>
#include <optional>
#include "cfm/cfm_attention.h"
#include "cfm/mdp_backbone.h"
#include "common/types.h"

namespace cms {
namespace cfm {

struct CFMInferenceConfig {
    std::filesystem::path manifest_path;
    std::filesystem::path model_path;
    std::filesystem::path output_dir;
    int    num_disparities = 96;
    double depth_min = 1.0;
    double depth_max = 80.0;
    int    max_samples = 0;
    bool   use_gpu = false;

    // New (Req 3.8, 4.10) — validated against metadata
    std::optional<AttentionMode>   attention_mode;
    std::optional<MDPBackboneType> mdp_backbone;
    std::filesystem::path mdp_backbone_weights;
    std::optional<int> num_heads;
};

class CFMInference {
public:
    /// Run CFM inference over the manifest and save predicted depth/disparity.
    /// Outputs `{sample_id}_depth.png` (uint16, meters × 256) and
    /// `{sample_id}_disp.png` (uint16, disparity × 256 — for evaluation with
    /// the existing cms evaluate command).
    static Result<void> run(const CFMInferenceConfig& config);
};

} // namespace cfm
} // namespace cms

#endif // CMS_HAS_LIBTORCH
