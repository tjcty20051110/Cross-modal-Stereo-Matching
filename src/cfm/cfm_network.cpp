#include "cfm/cfm_network.h"

#ifdef CMS_HAS_LIBTORCH

#include "cfm/mdp_backbone.h"

namespace cms {
namespace cfm {

// ---------------------------------------------------------------------------
// ConvBnReLU
// ---------------------------------------------------------------------------

ConvBnReLUImpl::ConvBnReLUImpl(int in_c, int out_c, int k, int s, int p) {
    conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_c, out_c, k)
                                   .stride(s)
                                   .padding(p)
                                   .bias(false));
    bn_   = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_c));
    register_module("conv", conv_);
    register_module("bn", bn_);
}

torch::Tensor ConvBnReLUImpl::forward(torch::Tensor x) {
    return torch::relu(bn_(conv_(x)));
}

// ---------------------------------------------------------------------------
// FeatureExtractor (PSMNet-style, downsample to H/4)
// ---------------------------------------------------------------------------

FeatureExtractorImpl::FeatureExtractorImpl() {
    stem1_  = ConvBnReLU(1,  16, 3, 1, 1);   // full res
    stem2_  = ConvBnReLU(16, 32, 3, 2, 1);   // H/2
    stem3_  = ConvBnReLU(32, 64, 3, 2, 1);   // H/4
    block1_ = ConvBnReLU(64, 64, 3, 1, 1);
    block2_ = ConvBnReLU(64, 64, 3, 1, 1);
    head_   = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, FEATURE_CHANNELS, 1)
                                      .stride(1)
                                      .bias(true));
    register_module("stem1", stem1_);
    register_module("stem2", stem2_);
    register_module("stem3", stem3_);
    register_module("block1", block1_);
    register_module("block2", block2_);
    register_module("head", head_);
}

torch::Tensor FeatureExtractorImpl::forward(torch::Tensor x) {
    x = stem1_->forward(x);
    x = stem2_->forward(x);
    x = stem3_->forward(x);
    auto res = x;
    x = block1_->forward(x);
    x = block2_->forward(x);
    x = x + res;  // residual
    x = head_(x);
    return x;
}

// ---------------------------------------------------------------------------
// CrossAttention — moved to cfm_attention.cpp (Task 2.2)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CFMModule
// ---------------------------------------------------------------------------

CFMModuleImpl::CFMModuleImpl(int feature_dim, int num_disparities,
                             AttentionMode attention_mode, int num_heads)
    : feature_dim_(feature_dim), num_disparities_(num_disparities),
      attention_mode_(attention_mode), num_heads_(num_heads) {
    feat_extractor_vis_ = FeatureExtractor();
    feat_extractor_thr_ = FeatureExtractor();

    // Strategy dispatch: construct the appropriate attention implementation.
    auto make_attn = [&]() -> std::shared_ptr<CrossAttentionImpl> {
        if (attention_mode == AttentionMode::LegacyGated) {
            return std::make_shared<LegacyGatedAttentionImpl>(feature_dim);
        }
        return std::make_shared<RealScaledDotAttentionImpl>(
            feature_dim, num_heads, /*downsample_factor=*/2);
    };
    cross_attn_vis_to_thr_ = CrossAttention(make_attn());
    cross_attn_thr_to_vis_ = CrossAttention(make_attn());

    register_module("feat_extractor_vis", feat_extractor_vis_);
    register_module("feat_extractor_thr", feat_extractor_thr_);
    register_module("cross_attn_vis_to_thr", cross_attn_vis_to_thr_);
    register_module("cross_attn_thr_to_vis", cross_attn_thr_to_vis_);
}

CFMOutput CFMModuleImpl::forward(torch::Tensor vis, torch::Tensor thr) {
    auto f_vis = feat_extractor_vis_->forward(vis);  // [B, C, H/4, W/4]
    auto f_thr = feat_extractor_thr_->forward(thr);

    // Align feature spaces via cross-attention (both directions, per paper)
    auto f_vis_aligned = cross_attn_thr_to_vis_->forward(f_vis, f_thr);
    auto f_thr_aligned = cross_attn_vis_to_thr_->forward(f_thr, f_vis);

    // Build cost volume by shifting visible features and computing cosine
    // similarity against thermal features at each disparity candidate.
    const int64_t B = f_thr_aligned.size(0);
    const int64_t C = f_thr_aligned.size(1);
    const int64_t H = f_thr_aligned.size(2);
    const int64_t W = f_thr_aligned.size(3);

    // L2-normalize along channel dim so dot product = cosine similarity
    auto fv_norm = torch::nn::functional::normalize(
        f_vis_aligned, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    auto ft_norm = torch::nn::functional::normalize(
        f_thr_aligned, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    // cost_volume[b, d, y, x] = sum_c ft[b, c, y, x] * fv[b, c, y, x - d]
    auto cost_volume = torch::zeros({B, num_disparities_, H, W},
                                     f_thr_aligned.options());
    for (int d = 0; d < num_disparities_; ++d) {
        // Shift fv_norm right by d pixels (zero-pad left)
        auto shifted = torch::zeros_like(fv_norm);
        if (d == 0) {
            shifted = fv_norm;
        } else if (d < W) {
            shifted.index({torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(),
                           torch::indexing::Slice(d, W)}) =
                fv_norm.index({torch::indexing::Slice(),
                               torch::indexing::Slice(),
                               torch::indexing::Slice(),
                               torch::indexing::Slice(0, W - d)});
        }
        cost_volume.index_put_(
            {torch::indexing::Slice(), d, torch::indexing::Slice(), torch::indexing::Slice()},
            (ft_norm * shifted).sum(1));
    }

    CFMOutput out;
    out.vis_feat = f_vis_aligned;
    out.thr_feat = f_thr_aligned;
    out.cost_volume = cost_volume;
    return out;
}

// ---------------------------------------------------------------------------
// MDPModule — refactored to use MDPBackbone strategy (Task 3.5)
// ---------------------------------------------------------------------------

MDPModuleImpl::MDPModuleImpl(int feature_dim, MDPBackboneType backbone_type,
                             const std::filesystem::path& pretrained_weights)
    : feature_dim_(feature_dim), backbone_type_(backbone_type) {

    // Construct the appropriate backbone.
    std::shared_ptr<MDPBackboneImpl> bb;
    switch (backbone_type) {
        case MDPBackboneType::LightweightCNN:
            bb = std::make_shared<LightweightCNNBackboneImpl>(feature_dim);
            break;
        case MDPBackboneType::ResNet18:
            bb = std::make_shared<LightweightCNNBackboneImpl>(feature_dim);
            // TODO(task 3.2): Replace with ResNet18BackboneImpl once implemented.
            break;
        case MDPBackboneType::MobileNetV3Small:
            bb = std::make_shared<LightweightCNNBackboneImpl>(feature_dim);
            // TODO(task 3.3): Replace with MobileNetV3SmallBackboneImpl once implemented.
            break;
    }
    backbone_ = bb;
    register_module("backbone", backbone_);

    // Load pretrained weights if provided.
    if (!pretrained_weights.empty() && std::filesystem::exists(pretrained_weights)) {
        bb->loadPretrained(pretrained_weights);
    } else if (!pretrained_weights.empty() &&
               backbone_type != MDPBackboneType::LightweightCNN) {
        std::cout << "[cfm-train] WARNING: no pretrained weights loaded for "
                  << mdpBackboneTypeToString(backbone_type)
                  << "; accuracy will be reduced" << std::endl;
    }

    head_mean_   = torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dim, 1, 1));
    head_logvar_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dim, 1, 1));
    register_module("head_mean", head_mean_);
    register_module("head_logvar", head_logvar_);
}

MDPOutput MDPModuleImpl::forward(torch::Tensor img) {
    // Forward through backbone.
    auto bb = std::dynamic_pointer_cast<MDPBackboneImpl>(backbone_);
    auto feat = bb->forward(img);

    auto mean    = head_mean_(feat);
    auto log_var = head_logvar_(feat).clamp(-6.0, 6.0);

    MDPOutput out;
    out.mean    = mean;
    out.log_var = log_var;
    out.feat    = feat;
    return out;
}

// ---------------------------------------------------------------------------
// Degradation Masking (paper Eq. 5-6)
// ---------------------------------------------------------------------------

torch::Tensor applyDegradationMask(const torch::Tensor& cost_volume,
                                    const MDPOutput& vis_mdp,
                                    const torch::Tensor& depth_candidates,
                                    double k_sigma) {
    // cost_volume: [B, D, H, W]
    // vis_mdp.mean:    [B, 1, H, W]
    // vis_mdp.log_var: [B, 1, H, W]
    // depth_candidates: [D]
    const int64_t D = cost_volume.size(1);

    auto sigma = (0.5 * vis_mdp.log_var).exp();   // [B, 1, H, W]
    auto mu    = vis_mdp.mean;                    // [B, 1, H, W]

    // Broadcast depth candidates to [1, D, 1, 1]
    auto d = depth_candidates.view({1, D, 1, 1}).to(cost_volume.device());

    // Gaussian probability P(d_k | (u,v)) — dropping the common 1/√(2πσ²)
    // factor (it cancels in the threshold comparison since threshold is also
    // computed from the same distribution).
    auto z  = (d - mu) / (sigma + 1e-6);          // [B, D, H, W]
    auto p  = torch::exp(-0.5 * z * z);           // [B, D, H, W], in (0, 1]

    // Threshold per pixel: max Gaussian density occurs at d = μ, value 1.
    // The paper keeps d_k if p(d_k) ≥ threshold. Since exp(−0.5·k²) is the
    // density at ±k·σ from μ, using k_sigma = 1 keeps only candidates within
    // ±1σ of μ. For soft masking we use a sigmoid to preserve gradients.
    const double threshold_density = std::exp(-0.5 * k_sigma * k_sigma);
    auto mask = torch::sigmoid(20.0 * (p - threshold_density));  // smooth gate ∈ (0,1)

    return cost_volume * mask;
}

// ---------------------------------------------------------------------------
// DepthModule
// ---------------------------------------------------------------------------

DepthModuleImpl::DepthModuleImpl(int num_disparities, int thermal_feat_dim,
                                  int upsample_factor)
    : upsample_factor_(upsample_factor) {
    const int in_c = num_disparities + thermal_feat_dim;
    fuse1_ = ConvBnReLU(in_c, 64, 3, 1, 1);
    fuse2_ = ConvBnReLU(64, 64, 3, 1, 1);
    fuse3_ = ConvBnReLU(64, 32, 3, 1, 1);
    head_mean_   = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 1, 1));
    head_logvar_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 1, 1));
    register_module("fuse1", fuse1_);
    register_module("fuse2", fuse2_);
    register_module("fuse3", fuse3_);
    register_module("head_mean", head_mean_);
    register_module("head_logvar", head_logvar_);
}

DepthOutput DepthModuleImpl::forward(torch::Tensor masked_cost,
                                       torch::Tensor thermal_feat) {
    auto x = torch::cat({masked_cost, thermal_feat}, /*dim=*/1);
    x = fuse1_->forward(x);
    x = fuse2_->forward(x);
    x = fuse3_->forward(x);

    auto mean    = head_mean_(x);
    auto log_var = head_logvar_(x).clamp(-6.0, 6.0);

    // Upsample to full resolution via bilinear interpolation
    auto H = mean.size(2) * upsample_factor_;
    auto W = mean.size(3) * upsample_factor_;
    mean = torch::nn::functional::interpolate(
        mean,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{H, W})
            .mode(torch::kBilinear)
            .align_corners(false));
    log_var = torch::nn::functional::interpolate(
        log_var,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{H, W})
            .mode(torch::kBilinear)
            .align_corners(false));

    DepthOutput out;
    out.mean    = mean;
    out.log_var = log_var;
    return out;
}

// ---------------------------------------------------------------------------
// CFMPipeline
// ---------------------------------------------------------------------------

CFMPipelineImpl::CFMPipelineImpl(int num_disparities, double depth_min,
                                   double depth_max,
                                   AttentionMode attention_mode,
                                   MDPBackboneType mdp_backbone,
                                   const std::filesystem::path& mdp_backbone_weights,
                                   int num_heads)
    : num_disparities_(num_disparities),
      depth_min_(depth_min),
      depth_max_(depth_max) {
    cfm_          = CFMModule(FeatureExtractorImpl::FEATURE_CHANNELS, num_disparities,
                              attention_mode, num_heads);
    vis_mdp_      = MDPModule(FeatureExtractorImpl::FEATURE_CHANNELS,
                              mdp_backbone, mdp_backbone_weights);
    thr_mdp_      = MDPModule(FeatureExtractorImpl::FEATURE_CHANNELS,
                              mdp_backbone, mdp_backbone_weights);
    depth_module_ = DepthModule(num_disparities,
                                 FeatureExtractorImpl::FEATURE_CHANNELS,
                                 FeatureExtractorImpl::DOWNSAMPLE_FACTOR);
    register_module("cfm", cfm_);
    register_module("vis_mdp", vis_mdp_);
    register_module("thr_mdp", thr_mdp_);
    register_module("depth_module", depth_module_);
}

torch::Tensor CFMPipelineImpl::depthCandidates(const torch::Device& device) const {
    // Uniform sampling in inverse-depth space (standard for cost volumes).
    auto inv_min = 1.0 / depth_max_;
    auto inv_max = 1.0 / depth_min_;
    auto inv = torch::linspace(inv_min, inv_max, num_disparities_,
                                torch::TensorOptions().dtype(torch::kFloat32).device(device));
    return 1.0 / inv;
}

CFMPipelineImpl::Output CFMPipelineImpl::forward(torch::Tensor vis,
                                                   torch::Tensor thr) {
    Output out;
    out.cfm = cfm_->forward(vis, thr);

    // MDP on each modality
    out.vis_mdp = vis_mdp_->forward(vis);
    out.thr_mdp = thr_mdp_->forward(thr);

    // Degradation masking
    auto depth_cands = depthCandidates(vis.device());
    auto masked_cost = applyDegradationMask(out.cfm.cost_volume, out.vis_mdp,
                                             depth_cands, /*k_sigma=*/1.0);

    // Depth regression
    out.final_depth = depth_module_->forward(masked_cost, out.thr_mdp.feat);
    return out;
}

} // namespace cfm
} // namespace cms

#endif // CMS_HAS_LIBTORCH
