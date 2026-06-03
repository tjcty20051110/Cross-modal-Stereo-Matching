// cfm_attention.cpp — Attention strategy implementations (Tasks 2.2, 2.3, 2.4).
#include "cfm/cfm_attention.h"

#ifdef CMS_HAS_LIBTORCH

#include <iostream>

namespace cms {
namespace cfm {

// ---------------------------------------------------------------------------
// Enum ↔ string helpers.
// ---------------------------------------------------------------------------

std::string attentionModeToString(AttentionMode mode) {
    switch (mode) {
        case AttentionMode::LegacyGated:   return "legacy_gated";
        case AttentionMode::RealScaledDot: return "real_scaled_dot";
    }
    return "legacy_gated";
}

std::optional<AttentionMode> attentionModeFromString(const std::string& s) {
    if (s == "legacy_gated")   return AttentionMode::LegacyGated;
    if (s == "real_scaled_dot") return AttentionMode::RealScaledDot;
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// LegacyGatedAttentionImpl — verbatim port of the original CrossAttention.
// ---------------------------------------------------------------------------

LegacyGatedAttentionImpl::LegacyGatedAttentionImpl(int feature_dim)
    : feature_dim_(feature_dim) {
    q_proj_   = torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dim, feature_dim, 1));
    k_proj_   = torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dim, feature_dim, 1));
    v_proj_   = torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dim, feature_dim, 1));
    out_proj_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dim, feature_dim, 1));
    register_module("q_proj", q_proj_);
    register_module("k_proj", k_proj_);
    register_module("v_proj", v_proj_);
    register_module("out_proj", out_proj_);
}

torch::Tensor LegacyGatedAttentionImpl::forward(torch::Tensor query_feat,
                                                  torch::Tensor key_feat) {
    const int64_t C = query_feat.size(1);
    auto Q = q_proj_(query_feat);
    auto K = k_proj_(key_feat);
    auto V = v_proj_(key_feat);

    // Channel-wise gated fusion (original approximation).
    auto score = torch::sigmoid((Q * K).sum(1, /*keepdim=*/true) /
                                  std::sqrt(static_cast<double>(C)));
    auto aligned = score * V + (1.0 - score) * query_feat;
    return out_proj_(aligned);
}

// ---------------------------------------------------------------------------
// RealScaledDotAttentionImpl — multi-head scaled dot-product (Req 3.1).
// ---------------------------------------------------------------------------

namespace {
constexpr int64_t kQueryChunkRows = 1024;
constexpr int64_t kMemoryBudgetBytes = 1LL * 1024 * 1024 * 1024;  // 1 GiB
}  // namespace

RealScaledDotAttentionImpl::RealScaledDotAttentionImpl(
    int feature_dim, int num_heads, int downsample_factor)
    : feature_dim_(feature_dim),
      num_heads_(num_heads),
      downsample_factor_(downsample_factor) {
    TORCH_CHECK(num_heads >= 2,
        "RealScaledDotAttention: num_heads must be >= 2, got ", num_heads);
    TORCH_CHECK(feature_dim % num_heads == 0,
        "RealScaledDotAttention: feature_dim (", feature_dim,
        ") must be divisible by num_heads (", num_heads, ")");
    d_head_ = feature_dim / num_heads;
    scale_  = 1.0 / std::sqrt(static_cast<double>(d_head_));

    q_proj_   = torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dim, feature_dim, 1));
    k_proj_   = torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dim, feature_dim, 1));
    v_proj_   = torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dim, feature_dim, 1));
    out_proj_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dim, feature_dim, 1));
    register_module("q_proj", q_proj_);
    register_module("k_proj", k_proj_);
    register_module("v_proj", v_proj_);
    register_module("out_proj", out_proj_);
}

torch::Tensor RealScaledDotAttentionImpl::forward(torch::Tensor q_feat,
                                                    torch::Tensor k_feat) {
    const int64_t B = q_feat.size(0);
    const int64_t C = q_feat.size(1);
    const int64_t H = q_feat.size(2);
    const int64_t W = q_feat.size(3);

    // 1×1 linear projections.
    auto Q = q_proj_(q_feat);
    auto K = k_proj_(k_feat);
    auto V = v_proj_(k_feat);

    // Downsample to reduce token count (Req 3.4).
    const int64_t Hd = H / downsample_factor_;
    const int64_t Wd = W / downsample_factor_;

    auto interp = [](torch::Tensor x, int64_t h, int64_t w) {
        return torch::nn::functional::interpolate(
            x, torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{h, w})
                    .mode(torch::kBilinear)
                    .align_corners(false));
    };

    auto Qd = interp(Q, Hd, Wd);
    auto Kd = interp(K, Hd, Wd);
    auto Vd = interp(V, Hd, Wd);

    // Reshape to multi-head: [B, num_heads, N, d_head].
    const int64_t N = Hd * Wd;
    auto to_heads = [&](torch::Tensor x) {
        return x.view({B, num_heads_, d_head_, N})
                .transpose(2, 3)
                .contiguous();
    };
    Qd = to_heads(Qd);
    Kd = to_heads(Kd);
    Vd = to_heads(Vd);

    // Scaled dot-product attention with chunked fallback.
    torch::Tensor attn_out;
    const int64_t attn_map_bytes = N * N * num_heads_ * 4;
    if (attn_map_bytes < kMemoryBudgetBytes) {
        // Single-shot.
        auto scores = torch::matmul(Qd, Kd.transpose(-2, -1)) * scale_;
        auto attn   = torch::softmax(scores, /*dim=*/-1);
        attn_out    = torch::matmul(attn, Vd);
    } else {
        // Chunked fallback (Req 3.4, 3.5).
        attn_out = torch::empty_like(Vd);
        for (int64_t start = 0; start < N; start += kQueryChunkRows) {
            int64_t end = std::min(start + kQueryChunkRows, N);
            auto q_chunk = Qd.index({torch::indexing::Slice(),
                                      torch::indexing::Slice(),
                                      torch::indexing::Slice(start, end),
                                      torch::indexing::Slice()});
            auto s_chunk = torch::matmul(q_chunk, Kd.transpose(-2, -1)) * scale_;
            auto a_chunk = torch::softmax(s_chunk, /*dim=*/-1);
            attn_out.index_put_(
                {torch::indexing::Slice(),
                 torch::indexing::Slice(),
                 torch::indexing::Slice(start, end),
                 torch::indexing::Slice()},
                torch::matmul(a_chunk, Vd));
        }
    }

    // Merge heads and upsample back.
    auto merged = attn_out.transpose(2, 3)
                           .contiguous()
                           .view({B, C, Hd, Wd});
    auto upsampled = interp(merged, H, W);
    return out_proj_(upsampled);
}

}  // namespace cfm
}  // namespace cms

#endif  // CMS_HAS_LIBTORCH
