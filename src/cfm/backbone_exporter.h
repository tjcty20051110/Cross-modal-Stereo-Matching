// backbone_exporter.h — Convert a TorchVision ImageNet `.pth` checkpoint into
// a LibTorch `torch::serialize::OutputArchive` file whose keys are exactly the
// subset consumed by `MDPBackbone` (Req 4.5, Req 4.7, Req 4.8).
//
// Purpose (task 4.2, design §3.2):
//   The project is pure C++ (no Python). Users manually download the official
//   TorchVision ImageNet checkpoints via `curl` — e.g.
//     resnet18-f37072fd.pth
//     mobilenet_v3_small-047dcff4.pth
//   — and pass the local `.pth` path to `cms export-mdp-backbone`. This
//   function:
//     1. Reads the raw bytes of the `.pth` file and decodes them via
//        `torch::pickle_load`, which yields an `IValue` dict of
//        `parameter_name → Tensor`.
//     2. Selects the subset of keys referenced by `MDPBackbone`:
//          ResNet18           → `conv1.*`, `bn1.*`, `layer1.*`
//          MobileNetV3-Small  → `features.0.*`, `features.1.*`
//        The `maxpool` layer has no learnable parameters, so no keys match it.
//     3. Applies the Grayscale_Adapter (Req 4.4) to the first convolution
//        weight: collapses the 3-channel input dimension to 1 via
//        `.mean(/*dim=*/1, /*keepdim=*/true)`.
//     4. Re-serializes the filtered dict as a LibTorch
//        `torch::serialize::OutputArchive` so that `loadPretrained` in a
//        future task can consume it with the standard `torch::load` machinery.
//
// The output file is *not* a TorchScript module — it is the flat archive
// format produced by `torch::save` for a tensor dict. That keeps the loader
// in `mdp_backbone.cpp` simple (no JIT, no IR execution).
#pragma once

#ifdef CMS_HAS_LIBTORCH

#include <filesystem>
#include <string>

#include "common/types.h"

namespace cms {
namespace cfm {

using Path = std::filesystem::path;

/// Export a TorchVision ImageNet checkpoint into an MDPBackbone-ready archive.
///
/// @param kind      One of `"resnet18"` or `"mobilenet_v3_small"` — any other
///                  value returns `Result<void>::error`.
/// @param pth_path  Path to a `.pth` file produced by TorchVision
///                  (e.g. `resnet18-f37072fd.pth`). Must be a plain pickled
///                  state_dict — the same format `torch::pickle_load` reads.
/// @param out_path  Where the filtered archive should be written. Parent
///                  directory is not auto-created; callers should mkdir first.
/// @return          `Result<void>` with an error message describing the
///                  failure (missing input, pickle decode error, empty
///                  filter match, archive save failure, …) or success on
///                  which stdout will have printed `saved: <out>` and
///                  `params: <N>`.
Result<void> exportMdpBackbone(const std::string& kind,
                               const Path& pth_path,
                               const Path& out_path);

}  // namespace cfm
}  // namespace cms

#endif  // CMS_HAS_LIBTORCH
