// cfm_network.h — Cross-modal Feature Matching network
// Reference: "Adaptive Stereo Depth Estimation with Multi-Spectral Images
//             Across All Lighting Conditions" (arXiv:2411.03638)
//
// This module is independent from the ResNet18 Siamese network in training/.
// It implements the four-stage pipeline:
//   1. CFM Module: cross-attention feature alignment + cost volume
//   2. MDP Module: per-modality Gaussian depth probability (μ, σ²)
//   3. Degradation Masking: mask cost volume by vis-MDP confidence
//   4. Depth Module: fuse masked cost + thermal features → depth + σ²
#pragma once

#ifdef CMS_HAS_LIBTORCH

#include <torch/torch.h>
#include <filesystem>

#include "cfm/cfm_attention.h"
#include "cfm/conv_bn_relu.h"
#include "cfm/mdp_backbone.h"

namespace cms {
namespace cfm {

// ---------------------------------------------------------------------------
// ConvBnReLU — defined in cfm/conv_bn_relu.h
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// FeatureExtractor — shared PSMNet-style backbone used inside CFM
// ---------------------------------------------------------------------------

/// Extracts a H/4 × W/4 × 32 feature map from a single-channel input image.
class FeatureExtractorImpl : public torch::nn::Module {
public:
    FeatureExtractorImpl();
    torch::Tensor forward(torch::Tensor x);

    static constexpr int FEATURE_CHANNELS = 32;
    static constexpr int DOWNSAMPLE_FACTOR = 4;

private:
    ConvBnReLU stem1_{nullptr};
    ConvBnReLU stem2_{nullptr};  // stride 2 → /2
    ConvBnReLU stem3_{nullptr};  // stride 2 → /4
    ConvBnReLU block1_{nullptr};
    ConvBnReLU block2_{nullptr};
    torch::nn::Conv2d head_{nullptr};
};
TORCH_MODULE(FeatureExtractor);

// ---------------------------------------------------------------------------
// CrossAttention — now declared in cfm/cfm_attention.h (polymorphic strategy)
// ---------------------------------------------------------------------------
// The concrete implementations (LegacyGatedAttentionImpl, RealScaledDotAttentionImpl)
// live in cfm_attention.{h,cpp}. CFMModuleImpl dispatches via the abstract base.

// ---------------------------------------------------------------------------
// CFMModule — paper Section III.B
// ---------------------------------------------------------------------------

/// Output of CFM Module.
struct CFMOutput {
    torch::Tensor vis_feat;    // [B, C, H/4, W/4]
    torch::Tensor thr_feat;    // [B, C, H/4, W/4]
    torch::Tensor cost_volume; // [B, D, H/4, W/4] — matching scores per disparity
};

/// Cross-modal Feature Matching Module.
/// Extracts features from both modalities, aligns them via cross-attention,
/// then builds a cost volume by shifted dot product over N disparity candidates.
class CFMModuleImpl : public torch::nn::Module {
public:
    CFMModuleImpl(int feature_dim = 32, int num_disparities = 96,
                  AttentionMode attention_mode = AttentionMode::RealScaledDot,
                  int num_heads = 4);

    /// @param vis  [B, 1, H, W] visible image (normalized)
    /// @param thr  [B, 1, H, W] thermal / NIR image (normalized)
    CFMOutput forward(torch::Tensor vis, torch::Tensor thr);

    int numDisparities() const { return num_disparities_; }

private:
    int feature_dim_;
    int num_disparities_;
    AttentionMode attention_mode_;
    int num_heads_;
    FeatureExtractor feat_extractor_vis_{nullptr};
    FeatureExtractor feat_extractor_thr_{nullptr};
    CrossAttention cross_attn_vis_to_thr_{nullptr};
    CrossAttention cross_attn_thr_to_vis_{nullptr};
};
TORCH_MODULE(CFMModule);

// ---------------------------------------------------------------------------
// MDPModule — paper Section III.C (Modality-specific Depth Probability)
// ---------------------------------------------------------------------------

/// Predicts per-pixel Gaussian depth distribution (μ, log_σ²) at H/4 × W/4.
struct MDPOutput {
    torch::Tensor mean;     // [B, 1, H/4, W/4]  (μ_uv)
    torch::Tensor log_var;  // [B, 1, H/4, W/4]  (log σ²_uv, for numerical stability)
    torch::Tensor feat;     // [B, C, H/4, W/4]  last-layer features (used by Depth Module)
};

class MDPModuleImpl : public torch::nn::Module {
public:
    MDPModuleImpl(int feature_dim = 32,
                  MDPBackboneType backbone_type = MDPBackboneType::LightweightCNN,
                  const std::filesystem::path& pretrained_weights = {});

    /// @param img  [B, 1, H, W]
    MDPOutput forward(torch::Tensor img);

    int featureChannels() const { return feature_dim_; }

private:
    int feature_dim_;
    MDPBackboneType backbone_type_;
    // Backbone is registered as a generic Module since we use the abstract base.
    std::shared_ptr<torch::nn::Module> backbone_{nullptr};
    torch::nn::Conv2d head_mean_{nullptr};
    torch::nn::Conv2d head_logvar_{nullptr};
};
TORCH_MODULE(MDPModule);

// ---------------------------------------------------------------------------
// Degradation Masking — paper Section III.C
// ---------------------------------------------------------------------------

/// Apply degradation mask to the cost volume using visible-modality MDP.
///
/// For each pixel (u,v) and depth candidate d_k:
///   threshold θ(u,v) = μ(u,v) + k·σ(u,v)
///   mask(d_k|u,v)    = 1 if P_vis(d_k|u,v) ≥ θ, else 0
/// Low-confidence entries are multiplied by ~0 so the cost volume contributes
/// nothing for those pixels; the Depth Module then falls back to thermal MDP
/// features in those regions (Eq. 6 in the paper).
///
/// @param cost_volume   [B, D, H/4, W/4]
/// @param vis_mdp       output of the visible-image MDP
/// @param depth_candidates  [D] tensor of depth values per disparity slot
/// @param k_sigma       threshold scale (paper uses k=1)
torch::Tensor applyDegradationMask(const torch::Tensor& cost_volume,
                                    const MDPOutput& vis_mdp,
                                    const torch::Tensor& depth_candidates,
                                    double k_sigma = 1.0);

// ---------------------------------------------------------------------------
// DepthModule — paper Section III.D
// ---------------------------------------------------------------------------

/// Fuses masked cost volume + thermal MDP features, predicts dense Gaussian
/// depth (μ, log_σ²) and learnably upsamples from H/4 to full resolution.
struct DepthOutput {
    torch::Tensor mean;     // [B, 1, H, W]
    torch::Tensor log_var;  // [B, 1, H, W]
};

class DepthModuleImpl : public torch::nn::Module {
public:
    DepthModuleImpl(int num_disparities, int thermal_feat_dim = 32,
                    int upsample_factor = 4);

    /// @param masked_cost    [B, D, H/4, W/4]
    /// @param thermal_feat   [B, C, H/4, W/4]
    DepthOutput forward(torch::Tensor masked_cost, torch::Tensor thermal_feat);

private:
    int upsample_factor_;
    ConvBnReLU fuse1_{nullptr};
    ConvBnReLU fuse2_{nullptr};
    ConvBnReLU fuse3_{nullptr};
    torch::nn::Conv2d head_mean_{nullptr};
    torch::nn::Conv2d head_logvar_{nullptr};
};
TORCH_MODULE(DepthModule);

// ---------------------------------------------------------------------------
// CFMPipeline — end-to-end wrapper
// ---------------------------------------------------------------------------

/// End-to-end wrapper composing CFM + two MDPs + Depth Module.
class CFMPipelineImpl : public torch::nn::Module {
public:
    CFMPipelineImpl(int num_disparities = 96, double depth_min = 1.0,
                    double depth_max = 80.0,
                    AttentionMode attention_mode = AttentionMode::RealScaledDot,
                    MDPBackboneType mdp_backbone = MDPBackboneType::LightweightCNN,
                    const std::filesystem::path& mdp_backbone_weights = {},
                    int num_heads = 4);

    /// Forward pass returning all intermediate results (useful for multi-stage
    /// loss computation and visualization).
    struct Output {
        CFMOutput  cfm;
        MDPOutput  vis_mdp;
        MDPOutput  thr_mdp;
        DepthOutput final_depth;
    };

    Output forward(torch::Tensor vis, torch::Tensor thr);

    /// Depth candidates used to build the cost volume (uniform in disparity
    /// space, equivalently inverse depth space).
    torch::Tensor depthCandidates(const torch::Device& device) const;

    int numDisparities() const { return num_disparities_; }
    double depthMin() const { return depth_min_; }
    double depthMax() const { return depth_max_; }

    // Public access for trainer (parameter groups, freezing stages)
    CFMModule   cfm_module()      const { return cfm_; }
    MDPModule   vis_mdp_module()  const { return vis_mdp_; }
    MDPModule   thr_mdp_module()  const { return thr_mdp_; }
    DepthModule depth_module()    const { return depth_module_; }

private:
    int num_disparities_;
    double depth_min_;
    double depth_max_;
    CFMModule cfm_{nullptr};
    MDPModule vis_mdp_{nullptr};
    MDPModule thr_mdp_{nullptr};
    DepthModule depth_module_{nullptr};
};
TORCH_MODULE(CFMPipeline);

} // namespace cfm
} // namespace cms

#endif // CMS_HAS_LIBTORCH
