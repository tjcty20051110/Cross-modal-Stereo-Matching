// cfm_mode_resolver.h — Resolve attention/backbone mode from checkpoint metadata + CLI flags.
#pragma once

#ifdef CMS_HAS_LIBTORCH

#include <filesystem>
#include <optional>
#include <string>

#include "cfm/cfm_attention.h"
#include "cfm/cfm_checkpoint_meta.h"
#include "cfm/mdp_backbone.h"
#include "common/types.h"

namespace cms {
namespace cfm {

struct ResolvedMode {
    AttentionMode    attention;
    MDPBackboneType  mdp_backbone;
    int              num_heads;
    int              feature_dim;
    int              num_disparities;
    double           depth_min;
    double           depth_max;
};

/// Resolve the pipeline construction parameters from a checkpoint's sidecar
/// metadata and optional CLI overrides.
///
/// @param model_path    Path to the `.pt` checkpoint file.
/// @param cli_attention Explicit `--attention` flag from CLI (nullopt if not given).
/// @param cli_backbone  Explicit `--mdp-backbone` flag from CLI (nullopt if not given).
/// @param cli_num_heads Explicit `--num-heads` flag from CLI (nullopt if not given).
/// @param ctx           Context string for log messages ("cfm-train" or "cfm-infer").
///
/// On success, returns the resolved mode. On error (mismatch, corrupt JSON),
/// returns an error message suitable for stderr.
Result<ResolvedMode> resolveMode(
    const std::filesystem::path& model_path,
    std::optional<AttentionMode> cli_attention,
    std::optional<MDPBackboneType> cli_backbone,
    std::optional<int> cli_num_heads,
    const std::string& ctx);

}  // namespace cfm
}  // namespace cms

#endif  // CMS_HAS_LIBTORCH
