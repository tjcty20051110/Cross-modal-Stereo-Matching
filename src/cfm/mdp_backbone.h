// mdp_backbone.h — strategy interface for the MDP encoder backbone.
//
// The MDP Module (paper Section III.C) consumes a single-channel image and
// produces a H/4 × W/4 × 32 feature map that feeds the per-pixel Gaussian
// depth heads (μ, log σ²) plus the DepthModule fusion stage. Historically
// `MDPModuleImpl` owned a 3-layer stem + 2-layer refine CNN inline; this
// header introduces a polymorphic backbone so the CNN can be swapped for an
// ImageNet-pretrained ResNet18 or MobileNetV3-Small without touching callers
// (Req 4.1, design §3.2).
//
// Output contract (Req 4.3):
//   forward(x: [B, 1, H, W]) → [B, 32, H/4, W/4]
//
// Registered submodule names inside `LightweightCNNBackboneImpl` are frozen
// at `stem1`, `stem2`, `stem3`, `refine1`, `refine2` to keep legacy `.pt`
// checkpoints deserializable key-for-key (Req 4.1, Req 5.6, Property 5).
#pragma once

#ifdef CMS_HAS_LIBTORCH

#include <torch/torch.h>

#include <filesystem>
#include <optional>
#include <string>

#include "cfm/conv_bn_relu.h"  // ConvBnReLU

namespace cms {
namespace cfm {

// ---------------------------------------------------------------------------
// MDPBackboneType — tri-state enum selecting the MDP encoder strategy.
// ---------------------------------------------------------------------------

enum class MDPBackboneType {
    LightweightCNN,    ///< Legacy 3-stem + 2-refine CNN (backward compat).
    ResNet18,          ///< ImageNet-pretrained ResNet18 (default, Req 4.1).
    MobileNetV3Small,  ///< ImageNet-pretrained MobileNetV3-Small.
};

/// Canonical string form for CLI flags, YAML configs, and checkpoint metadata.
/// Mirrors the enum names used in requirements.md: `lightweight_cnn`,
/// `resnet18`, `mobilenet_v3_small`.
std::string mdpBackboneTypeToString(MDPBackboneType type);

/// Parse `mdpBackboneTypeToString` output back into the enum. Returns
/// `std::nullopt` on any unknown token (case-sensitive match, same policy
/// as `attentionModeFromString`).
std::optional<MDPBackboneType> mdpBackboneTypeFromString(const std::string& s);

// ---------------------------------------------------------------------------
// MDPBackboneImpl — abstract strategy base.
// ---------------------------------------------------------------------------

/// Polymorphic base for all MDP encoder backbones. Subclasses must register
/// their learnable submodules so that `torch::save` / `torch::load` and the
/// stage-freeze logic in `CFMTrainer` see them through `named_parameters()`.
///
/// This is a `torch::nn::Module` (not a bare C++ interface) so that the
/// standard PyTorch C++ API — `parameters()`, `to(device)`, `train()`,
/// `eval()` — works uniformly regardless of which concrete backbone is
/// plugged in (design §3.2).
class MDPBackboneImpl : public torch::nn::Module {
public:
    /// @param x  single-channel input image `[B, 1, H, W]` with
    ///           `H % 4 == 0 && W % 4 == 0`.
    /// @return   feature map `[B, featureChannels(), H/4, W/4]`.
    virtual torch::Tensor forward(torch::Tensor x) = 0;

    /// Load pretrained weights from a TorchScript archive (ResNet18 /
    /// MobileNetV3-Small). Legacy CNN backbones treat this as a no-op.
    /// Must throw on any error (missing file, shape mismatch, missing key);
    /// silent fallback to random init is forbidden (Req 4.6).
    virtual void loadPretrained(const std::filesystem::path& path) = 0;

    /// Number of output channels at H/4. All current strategies return 32
    /// so `DepthModule` and `MDPOutput.feat` contracts stay stable
    /// (Req 4.3); override only if a future backbone changes the contract.
    virtual int featureChannels() const { return 32; }

    ~MDPBackboneImpl() override = default;
};

/// `torch::nn::ModuleHolder` wrapper around a shared pointer to the abstract
/// base. Callers construct the concrete subclass via `std::make_shared` and
/// wrap the result: `MDPBackbone bb(std::make_shared<ResNet18BackboneImpl>());`.
TORCH_MODULE(MDPBackbone);

// ---------------------------------------------------------------------------
// LightweightCNNBackboneImpl — legacy 3-stem + 2-refine CNN.
// ---------------------------------------------------------------------------

/// Verbatim port of the original `MDPModuleImpl` encoder (stem1 → stem2 → 
/// stem3 → refine1 → refine2 with a residual skip from stem3 output). The
/// heads (`head_mean`, `head_logvar`) stay on `MDPModuleImpl` so the
/// `MDPOutput` struct stays unchanged (Req 4.3).
///
/// Submodule names are frozen at `stem1`, `stem2`, `stem3`, `refine1`,
/// `refine2` to keep legacy `.pt` files loadable key-for-key (Req 4.1,
/// Req 5.6, Property 5).
class LightweightCNNBackboneImpl : public MDPBackboneImpl {
public:
    explicit LightweightCNNBackboneImpl(int feature_dim = 32);

    torch::Tensor forward(torch::Tensor x) override;

    /// No-op; emits a single log line on first call so users hitting
    /// `--mdp-backbone-weights` with the legacy backbone are not silently
    /// ignored. Returns immediately without touching any tensors.
    void loadPretrained(const std::filesystem::path& path) override;

    int featureChannels() const override { return feature_dim_; }

private:
    int feature_dim_;
    ConvBnReLU stem1_{nullptr};
    ConvBnReLU stem2_{nullptr};    // stride 2 → /2
    ConvBnReLU stem3_{nullptr};    // stride 2 → /4
    ConvBnReLU refine1_{nullptr};
    ConvBnReLU refine2_{nullptr};
};

/// Holder type for direct instantiation when the caller does not need the
/// abstract-base polymorphism (e.g. unit tests):
///   `LightweightCNNBackbone bb;`
TORCH_MODULE(LightweightCNNBackbone);

}  // namespace cfm
}  // namespace cms

#endif  // CMS_HAS_LIBTORCH
