#include "cfm/cfm_inference.h"

#ifdef CMS_HAS_LIBTORCH

#include <chrono>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

#include "cfm/cfm_mode_resolver.h"
#include "cfm/cfm_network.h"
#include "data/data_loader.h"
#include "preprocessing/preprocessor.h"

namespace cms {
namespace cfm {

Result<void> CFMInference::run(const CFMInferenceConfig& config) {
    DataLoader loader;
    auto count = loader.loadManifest(config.manifest_path);
    if (!count) return Result<void>::error(count.error_msg());
    const size_t total = config.max_samples > 0
                              ? std::min<size_t>(static_cast<size_t>(config.max_samples), loader.size())
                              : loader.size();
    if (total == 0) return Result<void>::error("No samples available");

    PipelineConfig preprocess_cfg;
    preprocess_cfg.enable_clahe = true;
    preprocess_cfg.enable_gradient_alignment = false;
    Preprocessor preprocessor(preprocess_cfg);

    torch::Device device(torch::kCPU);
    if (config.use_gpu && torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }

    if (!std::filesystem::exists(config.model_path)) {
        return Result<void>::error("Model file not found: " + config.model_path.string());
    }

    // Resolve mode from metadata (Req 3.8, 4.10, 5.4).
    auto resolved = resolveMode(config.model_path,
                                 config.attention_mode,
                                 config.mdp_backbone,
                                 config.num_heads,
                                 "cfm-infer");
    if (!resolved) {
        return Result<void>::error(resolved.error_msg());
    }
    auto mode = resolved.value();

    CFMPipeline network(mode.num_disparities > 0 ? mode.num_disparities : config.num_disparities,
                        mode.depth_min > 0 ? mode.depth_min : config.depth_min,
                        mode.depth_max > 0 ? mode.depth_max : config.depth_max,
                        mode.attention, mode.mdp_backbone,
                        config.mdp_backbone_weights, mode.num_heads);

    try {
        torch::load(network, config.model_path.string());
    } catch (const c10::Error& e) {
        std::string msg = e.what();
        if (msg.find("size mismatch") != std::string::npos ||
            msg.find("shape") != std::string::npos) {
            return Result<void>::error(
                std::string("Failed to load model (shape mismatch). ") +
                "请显式传入 --attention 与 --mdp-backbone. Detail: " + msg);
        }
        return Result<void>::error(std::string("Failed to load model: ") + msg);
    }

    network->to(device);
    network->eval();

    std::filesystem::create_directories(config.output_dir);

    const auto start = std::chrono::steady_clock::now();
    int success = 0, failed = 0;

    for (size_t i = 0; i < total; ++i) {
        auto sample = loader.getSample(i);
        if (!sample) { ++failed; continue; }

        auto prep = preprocessor.process(sample.value().left_image,
                                          sample.value().right_image,
                                          sample.value().left_modality,
                                          sample.value().right_modality);
        if (!prep) { ++failed; continue; }

        const cv::Mat& left  = prep.value().left;
        const cv::Mat& right = prep.value().right;

        cv::Mat lc = left.isContinuous() ? left : left.clone();
        cv::Mat rc = right.isContinuous() ? right : right.clone();
        auto thr_t = torch::from_blob(lc.data, {1, 1, lc.rows, lc.cols}, torch::kFloat32)
                          .clone().to(device);
        auto vis_t = torch::from_blob(rc.data, {1, 1, rc.rows, rc.cols}, torch::kFloat32)
                          .clone().to(device);

        CFMPipelineImpl::Output out;
        {
            torch::NoGradGuard no_grad;
            out = network->forward(vis_t, thr_t);
        }

        auto depth = out.final_depth.mean.squeeze(0).squeeze(0).to(torch::kCPU);
        auto depth_mat = cv::Mat(depth.size(0), depth.size(1), CV_32F,
                                   depth.data_ptr<float>()).clone();

        const double focal = sample.value().camera.focal_length;
        const double baseline = sample.value().camera.baseline;
        cv::Mat disp(depth_mat.size(), CV_32F, cv::Scalar(0));
        cv::Mat mask(depth_mat.size(), CV_8U, cv::Scalar(0));
        for (int y = 0; y < depth_mat.rows; ++y) {
            const float* drow = depth_mat.ptr<float>(y);
            float* disp_row = disp.ptr<float>(y);
            uint8_t* mrow = mask.ptr<uint8_t>(y);
            for (int x = 0; x < depth_mat.cols; ++x) {
                float d = drow[x];
                if (d > 0.0f && std::isfinite(d)) {
                    disp_row[x] = static_cast<float>(focal * baseline / d);
                    mrow[x] = 255;
                }
            }
        }

        const std::string sid = sample.value().sample_id;
        cv::Mat depth16;
        depth_mat.convertTo(depth16, CV_16U, 256.0);
        cv::imwrite((config.output_dir / (sid + "_depth.png")).string(), depth16);

        cv::Mat disp16;
        disp.convertTo(disp16, CV_16U, 256.0);
        cv::imwrite((config.output_dir / (sid + "_disp.png")).string(), disp16);
        cv::imwrite((config.output_dir / (sid + "_mask.png")).string(), mask);

        ++success;
    }

    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "[cfm-infer] success=" << success << " failed=" << failed
              << " elapsed=" << elapsed << " sec" << std::endl;
    return Result<void>::success();
}

}  // namespace cfm
}  // namespace cms

#endif  // CMS_HAS_LIBTORCH
