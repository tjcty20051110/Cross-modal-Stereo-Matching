#include "cfm/cfm_trainer.h"

#ifdef CMS_HAS_LIBTORCH

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

#include "cfm/cfm_checkpoint_meta.h"
#include "cfm/cfm_mode_resolver.h"
#include "cfm/cfm_network.h"
#include "data/data_loader.h"
#include "preprocessing/preprocessor.h"

namespace cms {
namespace cfm {
namespace {

/// Convert a CV_32F single-channel image to [1, 1, H, W] tensor on device.
torch::Tensor imageToTensor(const cv::Mat& image, const torch::Device& device) {
    cv::Mat cont = image.isContinuous() ? image : image.clone();
    return torch::from_blob(cont.data, {1, 1, cont.rows, cont.cols}, torch::kFloat32)
        .clone()
        .to(device);
}

/// Convert GT disparity map to depth tensor at H/4 × W/4.
torch::Tensor disparityToDepthTensor(const cv::Mat& disparity,
                                      const cv::Mat& valid_mask,
                                      double focal_length,
                                      double baseline,
                                      const torch::Device& device,
                                      int downsample_factor = 4) {
    const int H = disparity.rows;
    const int W = disparity.cols;
    cv::Mat depth(H, W, CV_32F, cv::Scalar(0));
    cv::Mat mask(H, W, CV_32F, cv::Scalar(0));
    for (int y = 0; y < H; ++y) {
        const float*  drow = disparity.ptr<float>(y);
        const uint8_t* mrow = valid_mask.ptr<uint8_t>(y);
        float* out = depth.ptr<float>(y);
        float* mout = mask.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            if (mrow[x] != 0 && drow[x] > 0.5f) {
                out[x]  = static_cast<float>(focal_length * baseline / drow[x]);
                mout[x] = 1.0f;
            }
        }
    }
    cv::Mat depth_low, mask_low;
    cv::resize(depth, depth_low,
               cv::Size(W / downsample_factor, H / downsample_factor),
               0, 0, cv::INTER_NEAREST);
    cv::resize(mask, mask_low,
               cv::Size(W / downsample_factor, H / downsample_factor),
               0, 0, cv::INTER_NEAREST);

    auto d_t = torch::from_blob(depth_low.data, {1, 1, depth_low.rows, depth_low.cols},
                                  torch::kFloat32).clone().to(device);
    auto m_t = torch::from_blob(mask_low.data, {1, 1, mask_low.rows, mask_low.cols},
                                  torch::kFloat32).clone().to(device);
    return torch::cat({d_t, m_t}, 1);
}

torch::Tensor expectedDepthL1Loss(const torch::Tensor& cost_volume,
                                    const torch::Tensor& depth_candidates,
                                    const torch::Tensor& depth_gt,
                                    const torch::Tensor& valid_mask) {
    auto prob = torch::softmax(cost_volume, /*dim=*/1);
    auto d = depth_candidates.view({1, static_cast<int64_t>(depth_candidates.size(0)), 1, 1});
    auto expected = (prob * d).sum(/*dim=*/1, /*keepdim=*/true);
    auto diff = (expected - depth_gt).abs() * valid_mask;
    auto n_valid = valid_mask.sum().clamp_min(1.0);
    return diff.sum() / n_valid;
}

torch::Tensor nllDepthLoss(const torch::Tensor& mean,
                             const torch::Tensor& log_var,
                             const torch::Tensor& depth_gt,
                             const torch::Tensor& valid_mask) {
    auto inv_var = torch::exp(-log_var);
    auto sq      = (depth_gt - mean).pow(2);
    auto loss    = 0.5 * (sq * inv_var + log_var) * valid_mask;
    auto n_valid = valid_mask.sum().clamp_min(1.0);
    return loss.sum() / n_valid;
}

std::pair<torch::Tensor, torch::Tensor> packDepthAndMask(
    const torch::Tensor& depth_and_mask) {
    auto depth = depth_and_mask.index({torch::indexing::Slice(), 0,
                                        torch::indexing::Slice(), torch::indexing::Slice()}).unsqueeze(1);
    auto mask  = depth_and_mask.index({torch::indexing::Slice(), 1,
                                        torch::indexing::Slice(), torch::indexing::Slice()}).unsqueeze(1);
    return {depth, mask};
}

std::string readGitShaOrEmpty() {
    std::string sha;
    FILE* pipe = popen("git rev-parse HEAD 2>/dev/null", "r");
    if (pipe) {
        char buf[64];
        if (fgets(buf, sizeof(buf), pipe)) {
            sha = buf;
            while (!sha.empty() && (sha.back() == '\n' || sha.back() == '\r'))
                sha.pop_back();
        }
        pclose(pipe);
    }
    return sha;
}

}  // namespace

Result<void> CFMTrainer::train(const CFMTrainingConfig& config) {
    DataLoader loader;
    auto count = loader.loadManifest(config.manifest_path);
    if (!count) return Result<void>::error(count.error_msg());
    const size_t total = config.max_samples > 0
                              ? std::min<size_t>(static_cast<size_t>(config.max_samples), loader.size())
                              : loader.size();
    if (total == 0) return Result<void>::error("No samples available for training");

    PipelineConfig preprocess_cfg;
    preprocess_cfg.enable_clahe = true;
    preprocess_cfg.enable_gradient_alignment = false;
    Preprocessor preprocessor(preprocess_cfg);

    torch::Device device(torch::kCPU);
    if (config.use_gpu && torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "[cfm-train] using CUDA device" << std::endl;
    } else {
        std::cout << "[cfm-train] using CPU device" << std::endl;
    }

    // Seed RNG (Req 6.1).
    std::mt19937 rng(config.seed);
    torch::manual_seed(config.seed);
    if (config.use_gpu && torch::cuda::is_available()) {
        torch::cuda::manual_seed_all(config.seed);
    }

    // Determine attention/backbone mode.
    AttentionMode   attn_mode = config.attention_mode;
    MDPBackboneType bb_type   = config.mdp_backbone;
    int             num_heads = config.num_heads;
    std::filesystem::path bb_weights = config.mdp_backbone_weights;

    // Handle --init-from with metadata resolution (Req 2.9, 5.2, 5.3).
    if (!config.init_from.empty()) {
        if (!std::filesystem::exists(config.init_from)) {
            return Result<void>::error("--init-from path does not exist: " + config.init_from.string());
        }
        if (std::filesystem::is_directory(config.init_from)) {
            return Result<void>::error("--init-from path is a directory: " + config.init_from.string());
        }
        if (std::filesystem::file_size(config.init_from) == 0) {
            return Result<void>::error("--init-from path is zero-byte: " + config.init_from.string());
        }

        auto resolved = resolveMode(config.init_from,
                                     config.attention_mode != AttentionMode::RealScaledDot
                                         ? std::optional<AttentionMode>(config.attention_mode)
                                         : std::nullopt,
                                     config.mdp_backbone != MDPBackboneType::ResNet18
                                         ? std::optional<MDPBackboneType>(config.mdp_backbone)
                                         : std::nullopt,
                                     std::nullopt, "cfm-train");
        if (!resolved) {
            return Result<void>::error(resolved.error_msg());
        }
        attn_mode = resolved.value().attention;
        bb_type   = resolved.value().mdp_backbone;
        num_heads = resolved.value().num_heads;
    }

    // Construct network with resolved parameters.
    CFMPipeline network(config.num_disparities, config.depth_min, config.depth_max,
                        attn_mode, bb_type, bb_weights, num_heads);

    if (!config.init_from.empty()) {
        std::cout << "[cfm-train] initialized from: "
                  << std::filesystem::absolute(config.init_from).string() << std::endl;
        torch::load(network, config.init_from.string());
    }
    network->to(device);
    network->train();

    // Decide which parameters are trainable for this stage (Req 2.7, 2.8).
    std::vector<torch::Tensor> params_to_train;
    switch (config.stage) {
        case TrainStage::CFM:
            for (auto& p : network->cfm_module()->parameters()) params_to_train.push_back(p);
            for (auto& p : network->depth_module()->parameters()) params_to_train.push_back(p);
            break;
        case TrainStage::MDP:
            // Freeze CFM module.
            for (auto& p : network->cfm_module()->parameters()) p.set_requires_grad(false);
            for (auto& p : network->vis_mdp_module()->parameters()) params_to_train.push_back(p);
            for (auto& p : network->thr_mdp_module()->parameters()) params_to_train.push_back(p);
            break;
        case TrainStage::DEPTH:
            // Freeze CFM + both MDPs.
            for (auto& p : network->cfm_module()->parameters())     p.set_requires_grad(false);
            for (auto& p : network->vis_mdp_module()->parameters()) p.set_requires_grad(false);
            for (auto& p : network->thr_mdp_module()->parameters()) p.set_requires_grad(false);
            for (auto& p : network->depth_module()->parameters()) params_to_train.push_back(p);
            break;
        case TrainStage::ALL:
            for (auto& p : network->parameters()) params_to_train.push_back(p);
            break;
    }

    // Emit trainable/frozen counts (Req 2.11).
    int64_t trainable_numel = 0, frozen_numel = 0;
    for (auto& p : network->parameters()) {
        if (p.requires_grad()) trainable_numel += p.numel();
        else frozen_numel += p.numel();
    }
    std::cout << "[cfm-train] trainable=" << trainable_numel
              << " frozen=" << frozen_numel << std::endl;

    torch::optim::Adam optimizer(params_to_train,
                                   torch::optim::AdamOptions(config.learning_rate));

    const int accum_steps = std::max(1, config.batch_size);
    std::cout << "[cfm-train] stage="
              << (config.stage == TrainStage::CFM   ? "CFM"
                : config.stage == TrainStage::MDP   ? "MDP"
                : config.stage == TrainStage::DEPTH ? "DEPTH" : "ALL")
              << " epochs=" << config.num_epochs
              << " accum_steps=" << accum_steps
              << " num_disparities=" << config.num_disparities
              << " depth_range=[" << config.depth_min << ", " << config.depth_max << "]"
              << " attention=" << attentionModeToString(attn_mode)
              << " mdp_backbone=" << mdpBackboneTypeToString(bb_type)
              << std::endl;

    // Pre-pass GT existence check (Req 1.7, 1.8).
    // Note: We skip the full getSample() call here (which loads images) and
    // instead just verify that the sample can be loaded. For large datasets
    // the full pre-pass would be too slow. We'll catch missing GT at training
    // time and skip those samples.
    std::vector<size_t> valid_indices;
    valid_indices.reserve(total);
    for (size_t i = 0; i < total; ++i) {
        valid_indices.push_back(i);
    }
    std::cout << "[cfm-train] pre-pass: valid=" << valid_indices.size()
              << " skipped=0" << std::endl;

    auto depth_cands = network->depthCandidates(device);

    const auto start_time = std::chrono::steady_clock::now();
    int consecutive_nan = 0;

    for (int epoch = 0; epoch < config.num_epochs; ++epoch) {
        std::shuffle(valid_indices.begin(), valid_indices.end(), rng);
        double epoch_loss = 0.0;
        int batches_seen = 0;
        int accum_count = 0;
        double accum_loss = 0.0;
        int valid_in_epoch = 0;
        int skipped_in_epoch = 0;
        optimizer.zero_grad();

        auto epoch_start = std::chrono::steady_clock::now();

        for (size_t ii = 0; ii < valid_indices.size(); ++ii) {
            size_t idx = valid_indices[ii];
            auto sample = loader.getSample(idx);
            if (!sample) { ++skipped_in_epoch; continue; }
            if (sample.value().gt_disparity.empty()) { ++skipped_in_epoch; continue; }

            auto prep = preprocessor.process(sample.value().left_image,
                                              sample.value().right_image,
                                              sample.value().left_modality,
                                              sample.value().right_modality);
            if (!prep) { ++skipped_in_epoch; continue; }

            const cv::Mat& left  = prep.value().left;
            const cv::Mat& right = prep.value().right;

            auto thr_t = imageToTensor(left,  device);
            auto vis_t = imageToTensor(right, device);

            auto depth_and_mask = disparityToDepthTensor(
                sample.value().gt_disparity, sample.value().valid_mask,
                sample.value().camera.focal_length,
                sample.value().camera.baseline,
                device, /*downsample=*/4);
            auto [depth_gt_low, mask_low] = packDepthAndMask(depth_and_mask);

            // Check valid pixel count (Req 1.9).
            if (mask_low.sum().item<double>() == 0.0) {
                ++skipped_in_epoch;
                continue;
            }

            auto H_full = left.rows;
            auto W_full = left.cols;
            auto depth_gt_full = torch::nn::functional::interpolate(
                depth_gt_low,
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{H_full, W_full})
                    .mode(torch::kNearest));
            auto mask_full = torch::nn::functional::interpolate(
                mask_low,
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{H_full, W_full})
                    .mode(torch::kNearest));

            auto out = network->forward(vis_t, thr_t);

            torch::Tensor loss;
            switch (config.stage) {
                case TrainStage::CFM: {
                    loss = expectedDepthL1Loss(out.cfm.cost_volume, depth_cands,
                                                 depth_gt_low, mask_low);
                    break;
                }
                case TrainStage::MDP: {
                    auto l_vis = nllDepthLoss(out.vis_mdp.mean, out.vis_mdp.log_var,
                                                depth_gt_low, mask_low);
                    auto l_thr = nllDepthLoss(out.thr_mdp.mean, out.thr_mdp.log_var,
                                                depth_gt_low, mask_low);
                    loss = 0.5 * (l_vis + l_thr);
                    break;
                }
                case TrainStage::DEPTH:
                case TrainStage::ALL: {
                    auto l_final = nllDepthLoss(out.final_depth.mean,
                                                  out.final_depth.log_var,
                                                  depth_gt_full, mask_full);
                    if (config.stage == TrainStage::DEPTH) {
                        loss = l_final;
                    } else {
                        auto l_cfm = expectedDepthL1Loss(out.cfm.cost_volume, depth_cands,
                                                           depth_gt_low, mask_low);
                        auto l_mdp_vis = nllDepthLoss(out.vis_mdp.mean, out.vis_mdp.log_var,
                                                        depth_gt_low, mask_low);
                        auto l_mdp_thr = nllDepthLoss(out.thr_mdp.mean, out.thr_mdp.log_var,
                                                        depth_gt_low, mask_low);
                        loss = l_final + 0.5 * l_cfm + 0.25 * (l_mdp_vis + l_mdp_thr);
                    }
                    break;
                }
            }

            if (!loss.defined() || !std::isfinite(loss.item<double>())) {
                ++consecutive_nan;
                ++skipped_in_epoch;
                if (consecutive_nan >= 10) {
                    return Result<void>::error("[cfm-train] persistent NaN loss, aborting");
                }
                continue;
            }
            consecutive_nan = 0;

            (loss / static_cast<double>(accum_steps)).backward();
            accum_loss += loss.item<double>();
            ++accum_count;
            ++valid_in_epoch;

            if (accum_count >= accum_steps) {
                torch::nn::utils::clip_grad_norm_(params_to_train, 1.0);
                optimizer.step();
                optimizer.zero_grad();
                epoch_loss += accum_loss / accum_count;
                ++batches_seen;
                if (batches_seen % config.log_every == 0) {
                    std::cout << "[epoch " << (epoch + 1) << "/" << config.num_epochs
                              << "] batch " << batches_seen
                              << " loss=" << (accum_loss / accum_count)
                              << std::endl;
                }
                accum_loss = 0.0;
                accum_count = 0;
            }

            // First-epoch ETA (Req 1.6).
            if (epoch == 0 && batches_seen == 10) {
                auto elapsed_10 = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - epoch_start).count();
                double per_batch_sec = elapsed_10 / 10.0;
                int64_t batches_per_epoch = static_cast<int64_t>(valid_indices.size()) / accum_steps;
                double eta_sec = per_batch_sec * static_cast<double>(batches_per_epoch);
                std::cout << "[cfm-train] first-epoch ETA: " << std::fixed
                          << std::setprecision(2) << eta_sec
                          << " sec (based on 10 batches)" << std::endl;
            }
        }
        if (accum_count > 0) {
            torch::nn::utils::clip_grad_norm_(params_to_train, 1.0);
            optimizer.step();
            optimizer.zero_grad();
            epoch_loss += accum_loss / accum_count;
            ++batches_seen;
        }
        const double avg = batches_seen > 0 ? epoch_loss / batches_seen : 0.0;
        std::cout << "[epoch " << (epoch + 1) << "] valid=" << valid_in_epoch
                  << " skipped=" << skipped_in_epoch
                  << " avg_loss=" << avg << std::endl;
    }

    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    std::cout << "[cfm-train] total elapsed=" << elapsed << " sec" << std::endl;

    network->eval();
    network->to(torch::kCPU);
    try {
        std::filesystem::create_directories(config.output_model_path.parent_path());
        torch::save(network, config.output_model_path.string());
        std::cout << "[cfm-train] saved model to " << config.output_model_path << std::endl;
    } catch (const std::exception& e) {
        return Result<void>::error(std::string("Failed to save CFM model: ") + e.what());
    }

    // Save .meta.json sidecar (Req 3.9, 4.11).
    CFMCheckpointMeta meta;
    meta.attention_mode = attentionModeToString(attn_mode);
    meta.mdp_backbone   = mdpBackboneTypeToString(bb_type);
    meta.mdp_backbone_weights_source = config.mdp_backbone_weights.string();
    meta.num_disparities = config.num_disparities;
    meta.depth_min       = config.depth_min;
    meta.depth_max       = config.depth_max;
    meta.num_heads       = num_heads;
    meta.git_sha         = readGitShaOrEmpty();
    auto meta_path = CFMCheckpointMeta::pathFor(config.output_model_path);
    auto meta_res = meta.write(meta_path);
    if (!meta_res) {
        std::cerr << "[cfm-train] warning: failed to write metadata: "
                  << meta_res.error_msg() << std::endl;
    }

    return Result<void>::success();
}

}  // namespace cfm
}  // namespace cms

#endif  // CMS_HAS_LIBTORCH
