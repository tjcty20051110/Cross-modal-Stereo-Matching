// siamese_network.h — ResNet18-inspired Siamese CNN for cross-modal features
#pragma once

#ifdef CMS_HAS_LIBTORCH

#include <torch/torch.h>

namespace cms {

// ---------------------------------------------------------------------------
// BasicBlock (identical to torchvision ResNet18's BasicBlock)
// ---------------------------------------------------------------------------

/// Residual block: conv3x3 → BN → ReLU → conv3x3 → BN → (+ skip) → ReLU
/// Optional dilation for enlarged receptive field without downsampling.
class BasicBlockImpl : public torch::nn::Module {
public:
    BasicBlockImpl(int in_channels, int out_channels, int stride = 1, int dilation = 1);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::Conv2d conv2_{nullptr};
    torch::nn::BatchNorm2d bn1_{nullptr};
    torch::nn::BatchNorm2d bn2_{nullptr};
    torch::nn::Sequential downsample_{nullptr};
    bool has_downsample_ = false;
};
TORCH_MODULE(BasicBlock);

// ---------------------------------------------------------------------------
// SiameseNetwork — ResNet18-inspired feature extractor
// ---------------------------------------------------------------------------

/// Dense cross-modal feature extractor inspired by ResNet18.
///
/// Architecture:
///
///   Input:  [B, 1, H, W]
///   Stem:   Conv 1→32, 3×3, BN, ReLU
///   Stage1: 2 × BasicBlock(32, 32,   stride=1, dilation=1)   → H × W
///   Stage2: 2 × BasicBlock(32→64,    stride=2, dilation=1)   → H/2 × W/2
///   Stage3: 2 × BasicBlock(64→128,   stride=1, dilation=2)   → H/2 × W/2
///   Stage4: 2 × BasicBlock(128→128,  stride=1, dilation=4)   → H/2 × W/2
///   Head:   Conv 128→32, 1×1
///   Upsample: bilinear → H × W
///   L2 normalize along channels
///   Output: [B, 32, H, W]
///
/// Depth: 1 stem + 8 BasicBlock (16 conv layers) + 1 head = ~18 conv layers,
/// mirroring ResNet18's layer count. Feature map is computed at H/2 to bound
/// memory (critical for 384×1224 inputs) and then bilinearly upsampled so the
/// stereo cost volume operates at full pixel resolution.
///
/// Effective receptive field ≈ 63 pixels (empirical), giving strong context
/// for cross-modal matching while keeping parameter count ~2.8M.
class SiameseNetworkImpl : public torch::nn::Module {
public:
    SiameseNetworkImpl();

    torch::Tensor forward(torch::Tensor x);

    /// Gather per-pixel feature vectors at integer (batch_idx, y, x) locations.
    torch::Tensor featuresAt(torch::Tensor feature_map, torch::Tensor yx);

private:
    torch::nn::Sequential makeStage(int in_channels, int out_channels,
                                     int num_blocks, int dilation, int stride = 1);

    torch::nn::Conv2d stem_conv_{nullptr};
    torch::nn::BatchNorm2d stem_bn_{nullptr};
    torch::nn::Sequential stage1_{nullptr};
    torch::nn::Sequential stage2_{nullptr};
    torch::nn::Sequential stage3_{nullptr};
    torch::nn::Sequential stage4_{nullptr};
    torch::nn::Conv2d head_{nullptr};
};
TORCH_MODULE(SiameseNetwork);

} // namespace cms

#endif // CMS_HAS_LIBTORCH
