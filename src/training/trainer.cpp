#include "training/trainer.h"

#ifdef CMS_HAS_LIBTORCH

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#include "preprocessing/preprocessor.h"
#include "training/siamese_network.h"

namespace cms {
namespace {

/// Convert a preprocessed CV_32F single-channel image to a [1, 1, H, W] tensor.
torch::Tensor imageToTensor(const cv::Mat& image, const torch::Device& device) {
    cv::Mat contiguous;
    if (image.isContinuous()) {
        contiguous = image;
    } else {
        contiguous = image.clone();
    }
    auto tensor = torch::from_blob(contiguous.data,
                                    {1, 1, contiguous.rows, contiguous.cols},
                                    torch::kFloat32)
                      .clone();
    return tensor.to(device);
}

/// Sample triplets (anchor, positive, negative) from one stereo pair.
/// anchor  = pixel in left feature map
/// positive = corresponding pixel in right feature map (x - disparity)
/// negative = pixel in right feature map at a random offset far from positive
struct TripletSamples {
    torch::Tensor anchors;    ///< [N, 3] (batch_idx, y, x)
    torch::Tensor positives;  ///< [N, 3] (batch_idx, y, x)
    torch::Tensor negatives;  ///< [N, 3] (batch_idx, y, x)
};

TripletSamples sampleTripletsForBatch(const std::vector<cv::Mat>& disparities,
                                       const std::vector<cv::Mat>& masks,
                                       int samples_per_image,
                                       int neg_min_offset,
                                       int neg_max_offset,
                                       int img_width,
                                       std::mt19937& rng) {
    std::vector<int64_t> anchor_rows, positive_rows, negative_rows;
    anchor_rows.reserve(samples_per_image * disparities.size() * 3);
    positive_rows.reserve(samples_per_image * disparities.size() * 3);
    negative_rows.reserve(samples_per_image * disparities.size() * 3);

    std::uniform_int_distribution<int> sign_dist(0, 1);

    for (size_t b = 0; b < disparities.size(); ++b) {
        const cv::Mat& disp = disparities[b];
        const cv::Mat& mask = masks[b];
        if (disp.empty() || mask.empty()) continue;

        // Collect valid pixel indices (where disparity > 0 and right match is inside image)
        std::vector<std::pair<int, int>> valid_yx;
        valid_yx.reserve(static_cast<size_t>(disp.rows * disp.cols) / 4);
        for (int y = 0; y < disp.rows; ++y) {
            const float* drow = disp.ptr<float>(y);
            const uint8_t* mrow = mask.ptr<uint8_t>(y);
            for (int x = 0; x < disp.cols; ++x) {
                if (mrow[x] == 0) continue;
                float d = drow[x];
                if (d <= 0.5f) continue;
                int x_match = x - static_cast<int>(std::round(d));
                if (x_match < 1 || x_match >= img_width - 1) continue;
                valid_yx.emplace_back(y, x);
            }
        }
        if (valid_yx.empty()) continue;

        std::uniform_int_distribution<size_t> pick(0, valid_yx.size() - 1);
        int collected = 0;
        int attempts = 0;
        while (collected < samples_per_image && attempts < samples_per_image * 4) {
            ++attempts;
            auto [y, x] = valid_yx[pick(rng)];
            float d = disp.at<float>(y, x);
            int x_pos = x - static_cast<int>(std::round(d));

            // Sample negative offset; avoid the region near the positive match
            std::uniform_int_distribution<int> offset_dist(neg_min_offset, neg_max_offset);
            int offset = offset_dist(rng);
            if (sign_dist(rng) == 0) offset = -offset;
            int x_neg = x_pos + offset;
            if (x_neg < 1 || x_neg >= img_width - 1) continue;

            anchor_rows.insert(anchor_rows.end(),
                               {static_cast<int64_t>(b), y, x});
            positive_rows.insert(positive_rows.end(),
                                  {static_cast<int64_t>(b), y, x_pos});
            negative_rows.insert(negative_rows.end(),
                                  {static_cast<int64_t>(b), y, x_neg});
            ++collected;
        }
    }

    TripletSamples triplets;
    const int64_t n = static_cast<int64_t>(anchor_rows.size() / 3);
    triplets.anchors = torch::from_blob(anchor_rows.data(), {n, 3}, torch::kLong).clone();
    triplets.positives = torch::from_blob(positive_rows.data(), {n, 3}, torch::kLong).clone();
    triplets.negatives = torch::from_blob(negative_rows.data(), {n, 3}, torch::kLong).clone();
    return triplets;
}

/// Preprocess a stereo sample pair into [H, W] CV_32F single-channel images
/// aligned to the reference (left) resolution.
Result<std::pair<cv::Mat, cv::Mat>> preprocessForTraining(
    const StereoSample& sample, const Preprocessor& preprocessor) {
    auto pp = preprocessor.process(sample.left_image, sample.right_image,
                                    sample.left_modality, sample.right_modality);
    if (!pp) {
        return Result<std::pair<cv::Mat, cv::Mat>>::error(pp.error_msg());
    }
    return Result<std::pair<cv::Mat, cv::Mat>>::success(
        std::make_pair(pp.value().left, pp.value().right));
}

} // namespace

Result<void> Trainer::train(const TrainingConfig& config) {
    // Load manifest
    DataLoader loader;
    auto count = loader.loadManifest(config.manifest_path);
    if (!count) return Result<void>::error(count.error_msg());
    const size_t total = config.max_samples > 0
                              ? std::min<size_t>(static_cast<size_t>(config.max_samples), loader.size())
                              : loader.size();
    if (total == 0) return Result<void>::error("No samples available for training");

    // Preprocessor mirrors the inference-time preprocessing
    PipelineConfig preprocess_cfg;
    preprocess_cfg.enable_clahe = true;
    preprocess_cfg.enable_gradient_alignment = false;
    Preprocessor preprocessor(preprocess_cfg);

    // Device selection
    torch::Device device(torch::kCPU);
    if (config.use_gpu && torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "[train] using CUDA device" << std::endl;
    } else {
        std::cout << "[train] using CPU device" << std::endl;
    }

    // Build network
    SiameseNetwork network;
    network->to(device);
    network->train();

    torch::optim::Adam optimizer(network->parameters(),
                                  torch::optim::AdamOptions(config.learning_rate));

    // batch_size is used as gradient accumulation steps.
    // Each step processes ONE image to keep GPU memory bounded, but the
    // optimizer only steps after `batch_size` images, simulating a larger batch.
    const int accum_steps = std::max(1, config.batch_size);
    std::cout << "[train] gradient accumulation steps=" << accum_steps
              << " (effective batch_size=" << accum_steps << ")" << std::endl;

    std::mt19937 rng(42);
    std::vector<size_t> indices(total);
    std::iota(indices.begin(), indices.end(), 0);

    const auto start_time = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < config.num_epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);
        double epoch_loss = 0.0;
        int batches_seen = 0;   // counts optimizer steps
        int accum_count = 0;    // counts images since last optimizer step
        double accum_loss = 0.0;

        optimizer.zero_grad();

        for (size_t i = 0; i < indices.size(); ++i) {
            auto sample = loader.getSample(indices[i]);
            if (!sample) continue;
            if (sample.value().gt_disparity.empty()) continue;
            auto prep = preprocessForTraining(sample.value(), preprocessor);
            if (!prep) continue;

            const cv::Mat& left  = prep.value().first;
            const cv::Mat& right = prep.value().second;

            // Process one image at a time (batch_size=1 for GPU memory)
            auto left_t  = imageToTensor(left,  device);  // [1, 1, H, W]
            auto right_t = imageToTensor(right, device);  // [1, 1, H, W]

            // Sample triplets for this single image (batch_idx always 0)
            std::vector<cv::Mat> disp_vec  = {sample.value().gt_disparity};
            std::vector<cv::Mat> mask_vec  = {sample.value().valid_mask};
            auto triplets = sampleTripletsForBatch(
                disp_vec, mask_vec, config.samples_per_image,
                config.neg_min_offset, config.neg_max_offset, left.cols, rng);
            if (triplets.anchors.size(0) == 0) continue;

            triplets.anchors   = triplets.anchors.to(device);
            triplets.positives = triplets.positives.to(device);
            triplets.negatives = triplets.negatives.to(device);

            auto left_feat  = network->forward(left_t);   // [1, C, H, W]
            auto right_feat = network->forward(right_t);  // [1, C, H, W]

            auto f_anchor   = network->featuresAt(left_feat,  triplets.anchors);
            auto f_positive = network->featuresAt(right_feat, triplets.positives);
            auto f_negative = network->featuresAt(right_feat, triplets.negatives);

            auto sim_pos = (f_anchor * f_positive).sum(1);
            auto sim_neg = (f_anchor * f_negative).sum(1);
            auto loss = torch::clamp_min(config.margin + sim_neg - sim_pos, 0.0).mean();

            // Scale loss by accumulation steps so effective gradient magnitude
            // matches a true batch of `accum_steps` images
            (loss / static_cast<double>(accum_steps)).backward();
            accum_loss += loss.item<double>();
            ++accum_count;

            // Optimizer step after accumulating `accum_steps` images
            if (accum_count >= accum_steps) {
                optimizer.step();
                optimizer.zero_grad();
                epoch_loss += accum_loss / accum_count;
                ++batches_seen;

                if (batches_seen % config.log_every == 0) {
                    std::cout << "[epoch " << (epoch + 1) << "/" << config.num_epochs
                              << "] batch " << batches_seen
                              << " loss=" << (accum_loss / accum_count)
                              << " triplets=" << (triplets.anchors.size(0) * accum_count)
                              << std::endl;
                }
                accum_loss  = 0.0;
                accum_count = 0;
            }
        }

        // Flush remaining accumulated gradients at epoch end
        if (accum_count > 0) {
            optimizer.step();
            optimizer.zero_grad();
            epoch_loss += accum_loss / accum_count;
            ++batches_seen;
        }

        const double avg = batches_seen > 0 ? epoch_loss / batches_seen : 0.0;
        std::cout << "[epoch " << (epoch + 1) << "] batches=" << batches_seen
                  << " avg_loss=" << avg << std::endl;
    }

    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    std::cout << "[train] total elapsed=" << elapsed << " sec" << std::endl;

    // Export model
    network->eval();
    network->to(torch::kCPU);
    try {
        std::filesystem::create_directories(config.output_model_path.parent_path());
        torch::save(network, config.output_model_path.string());
        std::cout << "[train] saved LibTorch model to " << config.output_model_path << std::endl;
    } catch (const std::exception& e) {
        return Result<void>::error(std::string("Failed to save model: ") + e.what());
    }

    return Result<void>::success();
}

} // namespace cms

#endif // CMS_HAS_LIBTORCH
