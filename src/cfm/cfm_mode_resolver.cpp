// cfm_mode_resolver.cpp — Implementation of resolveMode (Task 5.5).
#include "cfm/cfm_mode_resolver.h"

#ifdef CMS_HAS_LIBTORCH

#include <iostream>

namespace cms {
namespace cfm {

Result<ResolvedMode> resolveMode(
    const std::filesystem::path& model_path,
    std::optional<AttentionMode> cli_attention,
    std::optional<MDPBackboneType> cli_backbone,
    std::optional<int> cli_num_heads,
    const std::string& ctx) {

    auto meta_path = CFMCheckpointMeta::pathFor(model_path);
    auto meta_r = CFMCheckpointMeta::read(meta_path);

    CFMCheckpointMeta meta;
    bool meta_is_legacy = false;

    if (meta_r) {
        meta = meta_r.value();
    } else if (meta_r.error_msg() == "missing") {
        // Legacy checkpoint: defaults per Req 5.2 / 5.4.
        meta = CFMCheckpointMeta::legacyDefaults();
        meta_is_legacy = true;
        std::cout << "[" << ctx << "] legacy checkpoint, using defaults: "
                  << "attention=legacy_gated mdp-backbone=lightweight_cnn"
                  << std::endl;
    } else {
        // Corrupt metadata file — hard error.
        return Result<ResolvedMode>::error(meta_r.error_msg());
    }

    // Req 5.3: error on explicit-flag vs metadata conflict.
    if (cli_attention && attentionModeToString(*cli_attention) != meta.attention_mode) {
        return Result<ResolvedMode>::error(
            "attention mismatch: checkpoint=" + meta.attention_mode +
            " cli=" + attentionModeToString(*cli_attention));
    }
    if (cli_backbone && mdpBackboneTypeToString(*cli_backbone) != meta.mdp_backbone) {
        return Result<ResolvedMode>::error(
            "mdp-backbone mismatch: checkpoint=" + meta.mdp_backbone +
            " cli=" + mdpBackboneTypeToString(*cli_backbone));
    }

    // Req 5.2: log the override-from-checkpoint line in cfm-train only.
    if (ctx == "cfm-train" && !meta_is_legacy) {
        std::cout << "[cfm-train] override defaults from checkpoint: "
                  << "attention=" << meta.attention_mode
                  << " mdp-backbone=" << meta.mdp_backbone << std::endl;
    }

    // Parse the stored enum strings back.
    auto attn_opt = attentionModeFromString(meta.attention_mode);
    if (!attn_opt) {
        return Result<ResolvedMode>::error(
            "unknown attention_mode in metadata: " + meta.attention_mode);
    }
    auto bb_opt = mdpBackboneTypeFromString(meta.mdp_backbone);
    if (!bb_opt) {
        return Result<ResolvedMode>::error(
            "unknown mdp_backbone in metadata: " + meta.mdp_backbone);
    }

    ResolvedMode r;
    r.attention      = *attn_opt;
    r.mdp_backbone   = *bb_opt;
    r.num_heads      = cli_num_heads.value_or(meta.num_heads);
    r.feature_dim    = meta.feature_dim;
    r.num_disparities = meta.num_disparities;
    r.depth_min      = meta.depth_min;
    r.depth_max      = meta.depth_max;
    return Result<ResolvedMode>::success(r);
}

}  // namespace cfm
}  // namespace cms

#endif  // CMS_HAS_LIBTORCH
