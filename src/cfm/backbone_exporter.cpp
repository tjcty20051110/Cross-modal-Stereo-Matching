// backbone_exporter.cpp — task 4.2 implementation. See header for design
// overview.
//
// Why `torch::pickle_load` rather than `torch::jit::load`?
//   The TorchVision `.pth` files published on PyTorch Hub are *pickled
//   state_dicts*, not TorchScript archives. They have no model code, no
//   graph, no forward function — just a `collections.OrderedDict` of
//   `str → Tensor`. `torch::jit::load` expects a module (with a forward);
//   `torch::pickle_load` handles the raw pickle, which is what we need.
//
// Why filter here instead of at load time in `MDPBackbone::loadPretrained`?
//   The `.pth` files are ~45 MB (ResNet18) and ~10 MB (MNV3-Small). Stripping
//   out the unused classifier/fc/layer[2-4] tensors keeps the checked-in
//   `resnet18_in1k.pt` small, and keeps `loadPretrained` from having to
//   branch on "is this a torchvision checkpoint?".
//
// Why the Grayscale_Adapter lives in the exporter, not the loader?
//   The exporter is the one place in the pipeline that knows "this is a
//   3-channel ImageNet weight". Once we save the collapsed `[out, 1, k, k]`
//   tensor, the loader just does an exact-shape `copy_` and never has to
//   think about channel adaptation again. This also lets a future exporter
//   for 1-channel inputs (e.g. pretrained SAR models) drop the `.mean(1)`
//   step cleanly.
#include "cfm/backbone_exporter.h"

#ifdef CMS_HAS_LIBTORCH

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <unordered_set>
#include <vector>

#include <torch/serialize.h>
#include <torch/torch.h>

namespace cms {
namespace cfm {
namespace {

/// Read a file into a `std::vector<char>`. Returns an empty vector and sets
/// `error_out` on any I/O failure (file missing, permission denied, read
/// short). `torch::pickle_load` takes exactly this type, so we stage into it.
std::vector<char> readFileBytes(const Path& path, std::string& error_out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        error_out = "cannot open file for reading: " + path.string();
        return {};
    }
    // Tell-based size query avoids std::distance on input iterators which
    // would consume the stream.
    f.seekg(0, std::ios::end);
    const std::streamsize size = f.tellg();
    if (size <= 0) {
        error_out = "file is empty or unseekable: " + path.string();
        return {};
    }
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(size));
    if (!f.read(buf.data(), size)) {
        error_out = "failed to read file: " + path.string();
        return {};
    }
    return buf;
}

/// Key predicate for ResNet18. Matches the set of parameter / buffer names
/// that `MDPBackbone::ResNet18` will consume (Req 4.5):
///     conv1.weight
///     bn1.{weight, bias, running_mean, running_var, num_batches_tracked}
///     layer1.0.*
///     layer1.1.*
/// All other keys (`layer2..4.*`, `fc.*`, `avgpool`, etc.) are dropped.
bool matchesResNet18(const std::string& key) {
    return key.rfind("conv1.",   0) == 0
        || key.rfind("bn1.",     0) == 0
        || key.rfind("layer1.",  0) == 0;
}

/// Key predicate for MobileNetV3-Small. Matches the first two blocks of the
/// `features` sequential (the stem conv + first InvertedResidual):
///     features.0.0.weight           (Conv2d)
///     features.0.1.*                (BatchNorm)
///     features.1.*                  (InvertedResidual sub-tree)
/// Everything from `features.2` onwards, plus classifier, is dropped.
bool matchesMobileNetV3Small(const std::string& key) {
    return key.rfind("features.0.", 0) == 0
        || key.rfind("features.1.", 0) == 0;
}

/// The name of the tensor that carries the 3-channel input conv weight. The
/// Grayscale_Adapter collapses this tensor's channel dim from 3→1 so the
/// MDPBackbone (which sees single-channel NIR / grayscale vis input) can
/// `copy_` it directly. Different for the two backbones — see design §3.2.
std::string firstConvKey(const std::string& kind) {
    if (kind == "resnet18") {
        return "conv1.weight";          // [64, 3, 7, 7]  → [64, 1, 7, 7]
    }
    if (kind == "mobilenet_v3_small") {
        return "features.0.0.weight";   // [16, 3, 3, 3]  → [16, 1, 3, 3]
    }
    return "";
}

}  // namespace

Result<void> exportMdpBackbone(const std::string& kind,
                               const Path& pth_path,
                               const Path& out_path) {
    // -----------------------------------------------------------------------
    // 1) Validate the `kind` argument and pick its key filter.
    // -----------------------------------------------------------------------
    bool (*matches)(const std::string&) = nullptr;
    if (kind == "resnet18") {
        matches = &matchesResNet18;
    } else if (kind == "mobilenet_v3_small") {
        matches = &matchesMobileNetV3Small;
    } else {
        return Result<void>::error(
            "unknown --kind: '" + kind +
            "' (expected 'resnet18' or 'mobilenet_v3_small')");
    }

    // -----------------------------------------------------------------------
    // 2) Read the `.pth` file into memory and decode it via pickle_load.
    //    The result is a generic dict of IValues — the TorchVision
    //    checkpoints we target wrap their state_dict in the top level.
    // -----------------------------------------------------------------------
    std::string io_err;
    std::vector<char> bytes = readFileBytes(pth_path, io_err);
    if (bytes.empty()) {
        return Result<void>::error(io_err);
    }

    c10::IValue loaded;
    try {
        loaded = torch::pickle_load(bytes);
    } catch (const std::exception& e) {
        return Result<void>::error(
            std::string("failed to pickle_load '") + pth_path.string() +
            "': " + e.what());
    }

    if (!loaded.isGenericDict()) {
        // We specifically need a dict at the top level. TorchVision's
        // published ImageNet checkpoints are pickled OrderedDicts, which
        // deserialize to `c10::Dict<IValue, IValue>` here. If a user points
        // us at a full `torch.save(model)` archive, it would be a Module —
        // handled by `torch::jit::load`, not this path.
        return Result<void>::error(
            "pickled value is not a dict — got tag=" +
            std::string(loaded.tagKind()) +
            ". Expected a TorchVision state_dict (OrderedDict of "
            "name→Tensor). Point --pth at e.g. resnet18-f37072fd.pth.");
    }

    auto dict = loaded.toGenericDict();

    // -----------------------------------------------------------------------
    // 3) Walk every entry, keep only the keys our MDPBackbone consumes,
    //    apply the Grayscale_Adapter to the first conv weight.
    // -----------------------------------------------------------------------
    const std::string first_conv = firstConvKey(kind);

    torch::serialize::OutputArchive archive;
    int kept = 0;
    std::unordered_set<std::string> seen_keys;

    for (const auto& entry : dict) {
        if (!entry.key().isString()) {
            // Non-string keys would be unusual in a state_dict; skip them
            // rather than fail so a user-packaged `.pth` with trailing
            // metadata doesn't break the export.
            continue;
        }
        const std::string key = entry.key().toStringRef();
        if (!matches(key)) {
            continue;
        }
        if (!entry.value().isTensor()) {
            // Same reasoning — tolerate unexpected non-tensor entries.
            continue;
        }
        torch::Tensor tensor = entry.value().toTensor();

        // Grayscale_Adapter (Req 4.4). Only the *first* conv weight has
        // `[out, 3, k, k]` shape; all other tensors in the selected subset
        // are either 1-D (bias / BN params / running stats), 4-D but with
        // a non-3 input channel count (depthwise / pointwise inside
        // InvertedResidual), or scalars (num_batches_tracked). We trigger
        // the mean only for the designated key and only when its shape
        // actually matches the expected `[*, 3, *, *]` layout — otherwise
        // we pass the tensor through untouched and let a future loader
        // raise the shape mismatch with its exact expected shape.
        if (key == first_conv
            && tensor.dim() == 4
            && tensor.size(1) == 3) {
            // .mean keeps the original dtype (float32) and is safe on CPU.
            tensor = tensor.mean(/*dim=*/1, /*keepdim=*/true).contiguous();
        }

        // Detach so we don't carry any (meaningless-on-weights) autograd
        // history through serialization; clone so the archive owns storage
        // independent of the pickle buffer we read above.
        torch::Tensor out_tensor = tensor.detach().clone();

        // `is_buffer=false` here regardless of whether the source was a
        // buffer in the original nn.Module — our loader (task 3.4) walks
        // this archive as a flat name→tensor map via `try_read`, so the
        // buffer flag is informational only.
        archive.write(key, out_tensor, /*is_buffer=*/false);
        seen_keys.insert(key);
        ++kept;
    }

    if (kept == 0) {
        return Result<void>::error(
            "no matching keys found in '" + pth_path.string() +
            "' for --kind " + kind +
            ". Did you pass a non-TorchVision .pth or the wrong --kind?");
    }

    // Sanity: if the user asked for a kind whose first-conv weight is
    // mandatory but the checkpoint didn't contain it, warn loudly. We
    // don't fail — the loader will — but surfacing it here is friendlier.
    if (!first_conv.empty() && seen_keys.count(first_conv) == 0) {
        std::cerr << "[export-mdp-backbone] warning: first-conv weight '"
                  << first_conv
                  << "' not found in checkpoint; the resulting archive may "
                     "fail to load in MDPBackbone." << std::endl;
    }

    // -----------------------------------------------------------------------
    // 4) Save the filtered archive. `save_to(std::string)` writes atomically
    //    from the caller's perspective in practice (it buffers before
    //    flushing), but we don't rely on that — the sidecar `.meta.json`
    //    atomicity (task 5.6) is a separate concern.
    // -----------------------------------------------------------------------
    try {
        archive.save_to(out_path.string());
    } catch (const std::exception& e) {
        return Result<void>::error(
            std::string("failed to save archive to '") + out_path.string() +
            "': " + e.what());
    }

    // -----------------------------------------------------------------------
    // 5) User-visible summary lines per Req 4.8.
    // -----------------------------------------------------------------------
    std::cout << "saved: "  << out_path.string() << std::endl;
    std::cout << "params: " << kept            << std::endl;
    return Result<void>::success();
}

}  // namespace cfm
}  // namespace cms

#endif  // CMS_HAS_LIBTORCH
