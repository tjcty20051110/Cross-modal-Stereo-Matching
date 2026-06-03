// mdp_backbone.cpp — MDP encoder backbone strategy implementations.
//
// This TU only hosts the enum string helpers and the legacy
// `LightweightCNNBackboneImpl`. `ResNet18BackboneImpl` and
// `MobileNetV3SmallBackboneImpl` land in subsequent tasks (3.2, 3.3, 3.4).
//
// Staging note: `MDPModuleImpl` in `cfm_network.cpp` still owns its own
// stem/refine members verbatim. Task 3.5 will swap that body to forward into
// the strategy declared here. For now this file compiles on its own and
// produces a fully-registered lightweight CNN backbone that is
// bit-for-bit equivalent to the legacy inline implementation.
#include "cfm/mdp_backbone.h"

#ifdef CMS_HAS_LIBTORCH

#include <iostream>

namespace cms {
namespace cfm {

// ---------------------------------------------------------------------------
// Enum ↔ string helpers.
// ---------------------------------------------------------------------------

std::string mdpBackboneTypeToString(MDPBackboneType type) {
    switch (type) {
        case MDPBackboneType::LightweightCNN:
            return "lightweight_cnn";
        case MDPBackboneType::ResNet18:
            return "resnet18";
        case MDPBackboneType::MobileNetV3Small:
            return "mobilenet_v3_small";
    }
    // Unreachable unless the enum gains a new variant without this switch
    // being updated; keep the compiler happy without a default: clause so
    // `-Wswitch-enum` can catch such omissions.
    return "lightweight_cnn";
}

std::optional<MDPBackboneType> mdpBackboneTypeFromString(const std::string& s) {
    if (s == "lightweight_cnn") return MDPBackboneType::LightweightCNN;
    if (s == "resnet18")        return MDPBackboneType::ResNet18;
    if (s == "mobilenet_v3_small") return MDPBackboneType::MobileNetV3Small;
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// LightweightCNNBackboneImpl — verbatim port of the original MDP encoder.
// ---------------------------------------------------------------------------
//
// Architecture mirror (for reference — matches the pre-refactor
// `MDPModuleImpl` body inside `cfm_network.cpp`):
//
//   stem1   : ConvBnReLU(1 → 16,           3×3, s=1, p=1)   full-res
//   stem2   : ConvBnReLU(16 → 32,          3×3, s=2, p=1)   /2
//   stem3   : ConvBnReLU(32 → feature_dim, 3×3, s=2, p=1)   /4
//   refine1 : ConvBnReLU(feature_dim → feature_dim, 3×3, s=1, p=1)
//   refine2 : ConvBnReLU(feature_dim → feature_dim, 3×3, s=1, p=1)
//   feat    = refine2(refine1(stem3_out)) + stem3_out        ← residual
//
// Output shape: `[B, feature_dim, H/4, W/4]` for any `H, W` where
// `H % 4 == 0 && W % 4 == 0`.
// ---------------------------------------------------------------------------

LightweightCNNBackboneImpl::LightweightCNNBackboneImpl(int feature_dim)
    : feature_dim_(feature_dim) {
    stem1_   = ConvBnReLU(1, 16, 3, 1, 1);
    stem2_   = ConvBnReLU(16, 32, 3, 2, 1);              // /2
    stem3_   = ConvBnReLU(32, feature_dim, 3, 2, 1);     // /4
    refine1_ = ConvBnReLU(feature_dim, feature_dim, 3, 1, 1);
    refine2_ = ConvBnReLU(feature_dim, feature_dim, 3, 1, 1);

    // Registered names frozen verbatim (Req 4.1, Property 5):
    register_module("stem1", stem1_);
    register_module("stem2", stem2_);
    register_module("stem3", stem3_);
    register_module("refine1", refine1_);
    register_module("refine2", refine2_);
}

torch::Tensor LightweightCNNBackboneImpl::forward(torch::Tensor x) {
    x = stem1_->forward(x);
    x = stem2_->forward(x);
    x = stem3_->forward(x);
    auto res = x;                   // skip from /4
    x = refine1_->forward(x);
    x = refine2_->forward(x);
    return x + res;                 // residual, same shape as `res`
}

void LightweightCNNBackboneImpl::loadPretrained(
    const std::filesystem::path& path) {
    // The legacy CNN has no matching ImageNet checkpoint. Emit a single
    // informational line so users passing `--mdp-backbone-weights` with
    // `lightweight_cnn` are not silently ignored, then return. The path
    // argument is intentionally unused here.
    (void)path;
    std::cout << "[mdp-backbone] lightweight_cnn: loadPretrained is a no-op"
              << std::endl;
}

}  // namespace cfm
}  // namespace cms

#endif  // CMS_HAS_LIBTORCH
