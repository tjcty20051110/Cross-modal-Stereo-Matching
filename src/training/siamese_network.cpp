#include "training/siamese_network.h"

#ifdef CMS_HAS_LIBTORCH

namespace cms {

// ---------------------------------------------------------------------------
// BasicBlock
// ---------------------------------------------------------------------------

BasicBlockImpl::BasicBlockImpl(int in_channels, int out_channels, int stride, int dilation) {
    // First 3x3 conv: may change channels / spatial size
    conv1_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                                    .stride(stride)
                                    .padding(dilation)
                                    .dilation(dilation)
                                    .bias(false));
    bn1_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels));
    // Second 3x3 conv: preserves dimensions
    conv2_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                                    .stride(1)
                                    .padding(dilation)
                                    .dilation(dilation)
                                    .bias(false));
    bn2_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels));

    register_module("conv1", conv1_);
    register_module("bn1", bn1_);
    register_module("conv2", conv2_);
    register_module("bn2", bn2_);

    if (stride != 1 || in_channels != out_channels) {
        has_downsample_ = true;
        downsample_ = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1)
                                    .stride(stride)
                                    .bias(false)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)));
        register_module("downsample", downsample_);
    }
}

torch::Tensor BasicBlockImpl::forward(torch::Tensor x) {
    auto identity = has_downsample_ ? downsample_->forward(x) : x;
    auto out = torch::relu(bn1_(conv1_(x)));
    out = bn2_(conv2_(out));
    out = out + identity;
    return torch::relu(out);
}

// ---------------------------------------------------------------------------
// SiameseNetwork
// ---------------------------------------------------------------------------

torch::nn::Sequential SiameseNetworkImpl::makeStage(int in_channels, int out_channels,
                                                     int num_blocks, int dilation,
                                                     int stride) {
    torch::nn::Sequential stage;
    stage->push_back(BasicBlock(in_channels, out_channels, stride, dilation));
    for (int i = 1; i < num_blocks; ++i) {
        stage->push_back(BasicBlock(out_channels, out_channels, /*stride=*/1, dilation));
    }
    return stage;
}

SiameseNetworkImpl::SiameseNetworkImpl() {
    // Stem: 1 → 32
    stem_conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3)
                                        .stride(1)
                                        .padding(1)
                                        .bias(false));
    stem_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));

    // 4 stages × 2 BasicBlocks each = 8 BasicBlocks total (ResNet18 depth).
    // Use one stride-2 downsample in stage2 to bound memory; keep the rest
    // at H/2 resolution and expand receptive field via dilated convs.
    stage1_ = makeStage(32, 32, 2, /*dilation=*/1, /*stride=*/1);
    stage2_ = makeStage(32, 64, 2, /*dilation=*/1, /*stride=*/2);
    stage3_ = makeStage(64, 128, 2, /*dilation=*/2, /*stride=*/1);
    stage4_ = makeStage(128, 128, 2, /*dilation=*/4, /*stride=*/1);

    // Head: 128 → 32 (1x1 conv). Output is upsampled back to input resolution
    // in forward() via bilinear interpolation to restore dense pixel matching.
    head_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 32, 1).stride(1).bias(true));

    register_module("stem_conv", stem_conv_);
    register_module("stem_bn", stem_bn_);
    register_module("stage1", stage1_);
    register_module("stage2", stage2_);
    register_module("stage3", stage3_);
    register_module("stage4", stage4_);
    register_module("head", head_);
}

torch::Tensor SiameseNetworkImpl::forward(torch::Tensor x) {
    const auto input_h = x.size(2);
    const auto input_w = x.size(3);
    x = torch::relu(stem_bn_(stem_conv_(x)));
    x = stage1_->forward(x);
    x = stage2_->forward(x);  // → H/2, W/2
    x = stage3_->forward(x);
    x = stage4_->forward(x);
    x = head_(x);
    // Upsample feature map back to input spatial resolution so downstream cost
    // volume builder sees pixel-aligned features (downsample_factor reported as 1).
    x = torch::nn::functional::interpolate(
        x, torch::nn::functional::InterpolateFuncOptions()
               .size(std::vector<int64_t>{input_h, input_w})
               .mode(torch::kBilinear)
               .align_corners(false));
    x = torch::nn::functional::normalize(
        x, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1).eps(1e-8));
    return x;
}

torch::Tensor SiameseNetworkImpl::featuresAt(torch::Tensor feature_map,
                                              torch::Tensor yx) {
    // feature_map: [B, C, H, W], yx: [N, 3] (batch_idx, y, x)
    // Returns: [N, C]
    // Uses grid_sample for memory-efficient differentiable sampling.
    const int64_t H = feature_map.size(2);
    const int64_t W = feature_map.size(3);
    const int64_t C = feature_map.size(1);
    const int64_t B = feature_map.size(0);

    auto batch_idx = yx.select(1, 0).to(torch::kLong);
    auto y_coords  = yx.select(1, 1).to(torch::kFloat32);
    auto x_coords  = yx.select(1, 2).to(torch::kFloat32);

    // Normalize coordinates to [-1, 1] for grid_sample
    // grid_sample expects (x, y) in [-1, 1] where (-1,-1) = top-left
    auto x_norm = (x_coords / (static_cast<float>(W) - 1.0f)) * 2.0f - 1.0f;
    auto y_norm = (y_coords / (static_cast<float>(H) - 1.0f)) * 2.0f - 1.0f;

    // Process each batch element separately to avoid large intermediates
    auto result = torch::zeros({yx.size(0), C}, feature_map.options());
    for (int64_t b = 0; b < B; ++b) {
        auto mask    = (batch_idx == b);
        auto indices = mask.nonzero().squeeze(1);
        if (indices.numel() == 0) continue;

        auto xb = x_norm.index_select(0, indices);  // [Nb]
        auto yb = y_norm.index_select(0, indices);  // [Nb]

        // grid_sample input: [1, C, H, W], grid: [1, 1, Nb, 2]
        auto grid = torch::stack({xb, yb}, 1)
                        .unsqueeze(0)
                        .unsqueeze(0);  // [1, 1, Nb, 2]

        auto fm_b = feature_map[b].unsqueeze(0);  // [1, C, H, W]
        auto sampled = torch::nn::functional::grid_sample(
            fm_b, grid,
            torch::nn::functional::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kZeros)
                .align_corners(true));
        // sampled: [1, C, 1, Nb] -> [Nb, C]
        auto feats = sampled.squeeze(0).squeeze(1).t();  // [Nb, C]
        result.index_copy_(0, indices, feats);
    }
    return result;
}

} // namespace cms

#endif // CMS_HAS_LIBTORCH
