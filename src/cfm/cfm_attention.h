// cfm_attention.h — Cross-modal attention strategy (Requirement 3)
//
// Declares the attention-mode enum, an abstract `CrossAttentionImpl` base, and
// the two concrete implementations used by `CFMModuleImpl`:
//   - `LegacyGatedAttentionImpl` — the channel-wise sigmoid-gated fusion from
//     the pre-improvement code (Req 5.6 / Req 3.6 `legacy_gated` mode).
//   - `RealScaledDotAttentionImpl` — the real multi-head scaled dot-product
//     attention (Req 3.1 / Req 3.2).
//
// Only this header declares the polymorphic `CrossAttention` module wrapper.
// Concrete instances are created via `std::make_shared<...Impl>(...)` and
// wrapped at the call site with `CrossAttention{impl}`.
//
// -----------------------------------------------------------------------------
// MIGRATION NOTE (task 2.1 → 2.2):
//   The current `src/cfm/cfm_network.h` still holds the pre-refactor concrete
//   `CrossAttentionImpl` and its own `TORCH_MODULE(CrossAttention)` macro call.
//   To avoid a redefinition error, this header is NOT YET `#include`d from any
//   translation unit. Task 2.2 (move legacy impl to `cfm_attention.cpp`) will
//   delete the legacy declarations from `cfm_network.h` and add the `#include`
//   here. Until then, adding this header alone is safe because:
//     - CMake lists `.cpp` sources explicitly, so adding a header does not
//       create a new translation unit.
//     - No existing `.cpp` references `cfm/cfm_attention.h` (verify with
//       `grep -n '#include "cfm/cfm_attention.h"' src`).
//   If a future edit accidentally includes this header while the legacy
//   declarations still live in `cfm_network.h`, the compiler will flag the
//   duplicate `cms::cfm::CrossAttentionImpl` definition — that is the signal
//   to finish task 2.2.
// -----------------------------------------------------------------------------

#pragma once

#ifdef CMS_HAS_LIBTORCH

#include <torch/torch.h>

#include <cmath>
#include <memory>
#include <optional>
#include <string>

namespace cms {
namespace cfm {

// ---------------------------------------------------------------------------
// AttentionMode — strategy selector (Req 3.6, Req 3.7)
// ---------------------------------------------------------------------------

/// Which cross-attention implementation `CFMModuleImpl` should construct.
enum class AttentionMode {
    LegacyGated,   ///< sigmoid-gated channel-wise fusion (pre-improvement impl)
    RealScaledDot  ///< softmax(QKᵀ / √d) · V, multi-head (design §3.1)
};

/// Canonical string form used in `--attention` CLI flags and in the
/// `<model>.pt.meta.json` sidecar (design §3.4).
/// Mapping: LegacyGated → "legacy_gated", RealScaledDot → "real_scaled_dot".
std::string attentionModeToString(AttentionMode mode);

/// Inverse of `attentionModeToString`. Returns `std::nullopt` for any string
/// that is not an exact match of the canonical spellings above.
std::optional<AttentionMode> attentionModeFromString(const std::string& s);

// ---------------------------------------------------------------------------
// CrossAttentionImpl — abstract base (design §3.1)
// ---------------------------------------------------------------------------

/// Polymorphic cross-attention module. Subclasses must produce an output
/// tensor of the same `[B, C, H, W]` shape as `query_feat` (Req 3.3).
///
/// Subclasses inherit from both `torch::nn::Module` (via this base) and
/// register their learnable projections in their own constructors, so
/// `parameters()`, `to(device)`, `train()`, and `torch::save` / `torch::load`
/// all work uniformly across the polymorphic hierarchy (design §3.1 rationale).
class CrossAttentionImpl : public torch::nn::Module {
public:
    /// @param query_feat  [B, C, H, W] — features used as Q
    /// @param key_feat    [B, C, H, W] — features used as K and V
    /// @return aligned feature map [B, C, H, W]
    virtual torch::Tensor forward(torch::Tensor query_feat,
                                  torch::Tensor key_feat) = 0;

    ~CrossAttentionImpl() override = default;
};

// ---------------------------------------------------------------------------
// LegacyGatedAttentionImpl — pre-improvement sigmoid-gated fusion
// ---------------------------------------------------------------------------

/// Channel-wise sigmoid-gated fusion, preserved verbatim for `--attention
/// legacy_gated` and for deserializing legacy checkpoints (Req 5.6).
/// Submodule registered names are pinned to `q_proj`, `k_proj`, `v_proj`,
/// `out_proj` so legacy `.pt` files load without key remapping
/// (Correctness Property 5).
class LegacyGatedAttentionImpl : public CrossAttentionImpl {
public:
    explicit LegacyGatedAttentionImpl(int feature_dim);

    torch::Tensor forward(torch::Tensor query_feat,
                          torch::Tensor key_feat) override;

private:
    int feature_dim_;
    torch::nn::Conv2d q_proj_{nullptr};
    torch::nn::Conv2d k_proj_{nullptr};
    torch::nn::Conv2d v_proj_{nullptr};
    torch::nn::Conv2d out_proj_{nullptr};
};

// ---------------------------------------------------------------------------
// RealScaledDotAttentionImpl — multi-head scaled dot-product (Req 3.1, 3.2)
// ---------------------------------------------------------------------------

/// Real cross-attention: `softmax(Q·Kᵀ / √d_head) · V` with `num_heads` heads.
/// To bound memory for the default (88, 320) token grid, the forward pass
/// downsamples Q/K/V to H/(4·downsample_factor) × W/(4·downsample_factor)
/// before computing the attention map, then upsamples the result back to the
/// input spatial size (design §3.1 Memory strategy).
///
/// Constructor-level invariants (hard-enforced with `TORCH_CHECK` in the
/// implementation, Req 3.2):
///   - `num_heads >= 2`
///   - `feature_dim % num_heads == 0`  (no silent rounding; errors terminate
///     initialization, surfaced to the caller as a C++ exception.)
///
/// Submodule registered names are `q_proj`, `k_proj`, `v_proj`, `out_proj`
/// (1×1 `Conv2d`), matching the legacy module so checkpoint inspection tools
/// see the same canonical layout.
class RealScaledDotAttentionImpl : public CrossAttentionImpl {
public:
    /// @param feature_dim       input channel count C (must be divisible by `num_heads`)
    /// @param num_heads         number of attention heads (default 4, must be ≥ 2)
    /// @param downsample_factor factor applied to (H, W) before attention
    ///                          (default 2 ⇒ attention computed at H/8, W/8
    ///                          when input comes in at H/4, W/4).
    explicit RealScaledDotAttentionImpl(int feature_dim,
                                        int num_heads = 4,
                                        int downsample_factor = 2);

    torch::Tensor forward(torch::Tensor query_feat,
                          torch::Tensor key_feat) override;

    int featureDim() const { return feature_dim_; }
    int numHeads() const { return num_heads_; }
    int dHead() const { return d_head_; }
    int downsampleFactor() const { return downsample_factor_; }

private:
    int feature_dim_;
    int num_heads_;
    int d_head_;              ///< = feature_dim_ / num_heads_
    int downsample_factor_;   ///< ≥ 1; default 2 ⇒ attention at H/8
    double scale_;            ///< = 1.0 / sqrt(d_head_)

    torch::nn::Conv2d q_proj_{nullptr};
    torch::nn::Conv2d k_proj_{nullptr};
    torch::nn::Conv2d v_proj_{nullptr};
    torch::nn::Conv2d out_proj_{nullptr};
};

// ---------------------------------------------------------------------------
// TORCH_MODULE wrapper (design §3.1)
// ---------------------------------------------------------------------------
//
// Concrete instances are constructed via `std::make_shared<...Impl>(...)` and
// wrapped at the call site with `CrossAttention{impl}`. Example:
//
//   std::shared_ptr<CrossAttentionImpl> impl =
//       std::make_shared<RealScaledDotAttentionImpl>(32, /*num_heads=*/4);
//   CrossAttention attn{impl};
//
// The macro generates `class CrossAttention : public torch::nn::ModuleHolder<
// CrossAttentionImpl>` — i.e. a shared-ptr wrapper around the abstract base,
// not around any concrete subclass. This is the pattern recommended by the
// LibTorch docs for polymorphic submodules.
//
// NOTE: Currently `cfm_network.h` declares its own `CrossAttentionImpl` plus
// the matching `TORCH_MODULE(CrossAttention)` macro call. This header is not
// yet `#include`d anywhere, so the duplicate-declaration conflict is benign.
// Task 2.2 removes the legacy declarations from `cfm_network.h`.

TORCH_MODULE(CrossAttention);

} // namespace cfm
} // namespace cms

#endif // CMS_HAS_LIBTORCH
