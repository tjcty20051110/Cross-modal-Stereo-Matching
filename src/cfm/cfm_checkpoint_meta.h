// cfm_checkpoint_meta.h — Sidecar metadata for CFM `.pt` checkpoints
//
// Every `<model>.pt` saved by `CFMTrainer` is accompanied by a sibling
// `<model>.pt.meta.json` describing the architecture variants (attention mode,
// MDP backbone type, num_heads) and the training-time hyperparameters required
// to reconstruct the `CFMPipeline` before calling `torch::load`.
//
// Rationale (design §3.4): LibTorch's serialized state-dict does not carry
// structural information about polymorphic sub-modules (e.g. whether
// `CrossAttention` is `LegacyGatedAttentionImpl` or `RealScaledDotAttentionImpl`).
// The sidecar closes that gap so that `cms cfm-infer` and `cms cfm-train
// --init-from ...` can both reconstruct the correct pipeline, detect
// CLI-vs-metadata mismatches (Req 5.3), and fall back to legacy defaults
// when the sidecar is absent (Req 5.2, 5.4).
//
// Requirements: 3.9, 4.11, 5 (metadata infra).
#pragma once

#ifdef CMS_HAS_LIBTORCH

#include <filesystem>
#include <string>

#include "common/types.h"

namespace cms {
namespace cfm {

/// Sidecar metadata written alongside each CFM `.pt` checkpoint.
///
/// The on-disk representation is a JSON object keyed exactly as the member
/// names below. `attention_mode` and `mdp_backbone` carry the canonical
/// string encodings of their respective enums (e.g. "real_scaled_dot",
/// "resnet18") so the JSON is human-inspectable and survives enum renumbering.
struct CFMCheckpointMeta {
    /// Schema version. Bump when fields are added or renamed so readers can
    /// detect forward-incompatible files. Current schema is 1.
    int schema_version = 1;

    /// Canonical string for `AttentionMode` — one of:
    ///   "legacy_gated" | "real_scaled_dot".
    std::string attention_mode;

    /// Canonical string for `MDPBackboneType` — one of:
    ///   "lightweight_cnn" | "resnet18" | "mobilenet_v3_small".
    std::string mdp_backbone;

    /// Filesystem path passed to `--mdp-backbone-weights` at training time.
    /// Empty when the backbone was initialized from random weights (legacy
    /// `lightweight_cnn`) or when the user deliberately skipped pretrained
    /// loading. Recorded for reproducibility only; not consulted on load.
    std::string mdp_backbone_weights_source;

    /// Number of disparity bins in the CFM cost volume.
    int num_disparities = 0;

    /// Depth range used when mapping disparity → depth (meters).
    double depth_min = 0.0;
    double depth_max = 0.0;

    /// Feature-map channel count feeding `CrossAttention` and the cost volume.
    int feature_dim = 32;

    /// Number of attention heads for `RealScaledDotAttention`. Ignored by the
    /// legacy gated variant but still serialized for round-trip symmetry.
    int num_heads = 4;

    /// Result of `git rev-parse HEAD` at save time, or empty if unavailable.
    std::string git_sha;

    /// SHA-1 of the serialized `CFMTrainingConfig` (sorted keys). Recorded to
    /// detect accidental re-training on a stale config; NOT validated at load.
    std::string training_config_hash;

    /// Derive the sidecar path from a model `.pt` path by appending
    /// ".meta.json". Example:
    ///   pathFor("models/cfm.pt") → "models/cfm.pt.meta.json"
    static std::filesystem::path pathFor(const std::filesystem::path& model_path);

    /// Read and parse a sidecar JSON file from disk.
    ///
    /// Distinguishes two failure modes (per the design error-handling table):
    ///   * File does not exist → returns an error whose `error_msg()` equals
    ///     the literal "missing". Callers map this to "legacy checkpoint" and
    ///     substitute `legacyDefaults()` (Req 5.2, 5.4).
    ///   * File exists but is malformed JSON or fails schema validation →
    ///     returns a hard error describing the parse failure; callers should
    ///     abort rather than guess.
    static Result<CFMCheckpointMeta> read(const std::filesystem::path& meta_path);

    /// Write this struct to disk as pretty-printed JSON.
    ///
    /// Atomicity: serializes to `<meta_path>.tmp` first, then performs a
    /// `std::filesystem::rename` to `<meta_path>` so readers never observe a
    /// partially-written file.
    Result<void> write(const std::filesystem::path& meta_path) const;

    /// Default metadata for a legacy checkpoint with no sidecar (Req 5.4).
    /// Returns:
    ///   { schema_version=1, attention_mode="legacy_gated",
    ///     mdp_backbone="lightweight_cnn", num_heads=4, feature_dim=32 }
    /// with all other numeric fields left at their struct defaults.
    static CFMCheckpointMeta legacyDefaults();
};

} // namespace cfm
} // namespace cms

#endif // CMS_HAS_LIBTORCH
