#include "inference/dl_inference_engine.h"

#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#ifdef CMS_HAS_LIBTORCH
#include <torch/torch.h>
#include "training/siamese_network.h"
#endif

namespace cms {

struct DLInferenceEngine::Impl {
    cv::dnn::Net net;
    bool initialized = false;
#ifdef CMS_HAS_LIBTORCH
    SiameseNetwork torch_net{nullptr};
    torch::Device torch_device{torch::kCPU};
    bool torch_initialized = false;
#endif
};

namespace {
cv::Mat ensureFloatInput(const cv::Mat& input) {
    cv::Mat float_input;
    if (input.depth() == CV_32F) {
        float_input = input;
    } else {
        input.convertTo(float_input, CV_32F, input.depth() == CV_8U ? 1.0 / 255.0 : 1.0);
    }
    return float_input;
}

cv::Mat blobToFeatureMap(const cv::Mat& blob) {
    if (blob.empty()) return {};

    if (blob.dims == 4 && blob.size[0] == 1) {
        const int channels = blob.size[1];
        const int height = blob.size[2];
        const int width = blob.size[3];
        std::vector<cv::Mat> split_channels;
        split_channels.reserve(channels);
        for (int c = 0; c < channels; ++c) {
            split_channels.emplace_back(height, width, CV_32F,
                                        const_cast<float*>(blob.ptr<float>(0, c)));
        }
        cv::Mat merged;
        cv::merge(split_channels, merged);
        return merged.clone();
    }

    if (blob.dims == 2) {
        return blob.clone();
    }

    return blob.clone();
}
}  // namespace

DLInferenceEngine::DLInferenceEngine(const PipelineConfig& config)
    : config_(config), impl_(std::make_unique<Impl>()) {}

DLInferenceEngine::~DLInferenceEngine() = default;

Result<void> DLInferenceEngine::initialize() {
    if (impl_->initialized) {
        return Result<void>::success();
    }

#ifdef CMS_HAS_LIBTORCH
    if (config_.inference_backend == InferenceBackend::LIBTORCH) {
        if (config_.model_weights_path.empty()) {
            return Result<void>::error("DL model_weights_path is empty");
        }
        if (!std::filesystem::exists(config_.model_weights_path)) {
            return Result<void>::error("DL model file does not exist: " + config_.model_weights_path.string());
        }
        try {
            impl_->torch_device = (config_.use_gpu && torch::cuda::is_available())
                                       ? torch::Device(torch::kCUDA)
                                       : torch::Device(torch::kCPU);
            using_gpu_ = impl_->torch_device.is_cuda();

            impl_->torch_net = SiameseNetwork();
            torch::load(impl_->torch_net, config_.model_weights_path.string());
            impl_->torch_net->to(impl_->torch_device);
            impl_->torch_net->eval();

            // Self-test forward
            auto dummy = torch::zeros({1, 1, 32, 32}, torch::kFloat32).to(impl_->torch_device);
            {
                torch::NoGradGuard no_grad;
                (void)impl_->torch_net->forward(dummy);
            }
            impl_->torch_initialized = true;
            impl_->initialized = true;
            return Result<void>::success();
        } catch (const std::exception& e) {
            return Result<void>::error(std::string("LibTorch initialize failed: ") + e.what());
        }
    }
#endif

    if (config_.inference_backend != InferenceBackend::OPENCV_DNN) {
        return Result<void>::error("Only OPENCV_DNN and LIBTORCH backends are implemented in this build");
    }
    if (config_.model_weights_path.empty()) {
        return Result<void>::error("DL model_weights_path is empty");
    }
    if (!std::filesystem::exists(config_.model_weights_path)) {
        return Result<void>::error("DL model file does not exist: " + config_.model_weights_path.string());
    }

    try {
        impl_->net = cv::dnn::readNet(config_.model_weights_path.string());
        if (impl_->net.empty()) {
            return Result<void>::error("OpenCV DNN failed to load model: " + config_.model_weights_path.string());
        }

        impl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        impl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        using_gpu_ = false;

        cv::Mat dummy = cv::Mat::zeros(32, 32, CV_32F);
        cv::Mat blob = cv::dnn::blobFromImage(dummy, 1.0, dummy.size(), cv::Scalar(), false, false);
        impl_->net.setInput(blob);
        (void)impl_->net.forward();

        impl_->initialized = true;
        return Result<void>::success();
    } catch (const cv::Exception& e) {
        return Result<void>::error(std::string("OpenCV DNN initialize failed: ") + e.what());
    }
}

Result<InferenceResult> DLInferenceEngine::infer(const cv::Mat& input) const {
    if (!impl_->initialized) {
        return Result<InferenceResult>::error("DLInferenceEngine is not initialized");
    }
    if (input.empty()) {
        return Result<InferenceResult>::error("DLInferenceEngine received empty input");
    }

#ifdef CMS_HAS_LIBTORCH
    if (impl_->torch_initialized) {
        try {
            const cv::Mat float_input = ensureFloatInput(input);
            cv::Mat contiguous = float_input.isContinuous() ? float_input : float_input.clone();
            const auto start = std::chrono::steady_clock::now();

            auto tensor = torch::from_blob(contiguous.data,
                                            {1, 1, contiguous.rows, contiguous.cols},
                                            torch::kFloat32)
                              .clone()
                              .to(impl_->torch_device);

            torch::Tensor output;
            {
                torch::NoGradGuard no_grad;
                output = impl_->torch_net->forward(tensor);  // [1, C, H, W]
            }
            output = output.to(torch::kCPU).contiguous();
            const int channels = static_cast<int>(output.size(1));
            const int height = static_cast<int>(output.size(2));
            const int width = static_cast<int>(output.size(3));

            // Convert [1, C, H, W] -> cv::Mat [H, W, C]
            std::vector<cv::Mat> channel_mats;
            channel_mats.reserve(channels);
            for (int c = 0; c < channels; ++c) {
                cv::Mat chan(height, width, CV_32F,
                             output.index({0, c}).data_ptr<float>());
                channel_mats.emplace_back(chan.clone());
            }
            cv::Mat merged;
            cv::merge(channel_mats, merged);

            const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start);
            InferenceResult result;
            result.output = merged;
            result.elapsed_time = elapsed;
            return Result<InferenceResult>::success(std::move(result));
        } catch (const std::exception& e) {
            return Result<InferenceResult>::error(std::string("LibTorch inference failed: ") + e.what());
        }
    }
#endif

    try {
        const cv::Mat float_input = ensureFloatInput(input);
        const auto start = std::chrono::steady_clock::now();
        cv::Mat blob = cv::dnn::blobFromImage(float_input, 1.0, float_input.size(), cv::Scalar(), false, false);
        impl_->net.setInput(blob);
        cv::Mat output = impl_->net.forward();
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start);

        InferenceResult result;
        result.output = blobToFeatureMap(output);
        result.elapsed_time = elapsed;
        return Result<InferenceResult>::success(std::move(result));
    } catch (const cv::Exception& e) {
        return Result<InferenceResult>::error(std::string("OpenCV DNN inference failed: ") + e.what());
    }
}

bool DLInferenceEngine::isUsingGPU() const { return using_gpu_; }

}  // namespace cms
