#include "features/feature_extractor.h"

#include <algorithm>

#include "inference/dl_inference_engine.h"

namespace cms {
namespace {
int clampIndex(int value, int low, int high) {
    return std::max(low, std::min(value, high));
}
}  // namespace

CensusFeatureExtractor::CensusFeatureExtractor(int window_size) : window_size_(window_size) {}

Result<FeatureTensor> CensusFeatureExtractor::extract(const cv::Mat& image) const {
    if (image.empty()) {
        return Result<FeatureTensor>::error("FeatureExtractor received empty image");
    }
    if (image.type() != CV_32F) {
        return Result<FeatureTensor>::error("CensusFeatureExtractor expects CV_32F single-channel input");
    }

    const int radius = window_size_ / 2;
    const int channels = getFeatureChannels();
    cv::Mat features(image.rows, image.cols, CV_32FC(channels), cv::Scalar(0));

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            float center = image.at<float>(y, x);
            float* dst = features.ptr<float>(y, x);
            int channel = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int yy = clampIndex(y + dy, 0, image.rows - 1);
                    int xx = clampIndex(x + dx, 0, image.cols - 1);
                    float neighbor = image.at<float>(yy, xx);
                    dst[channel++] = neighbor > center ? 1.0f : 0.0f;
                }
            }
        }
    }

    FeatureTensor tensor;
    tensor.data = features;
    tensor.channels = channels;
    tensor.downsample_factor = 1;
    return Result<FeatureTensor>::success(tensor);
}

int CensusFeatureExtractor::getFeatureChannels() const {
    return window_size_ * window_size_ - 1;
}

DLFeatureExtractor::DLFeatureExtractor(const PipelineConfig& config) : config_(config) {
    if (config_.dl_fallback_to_handcraft) {
        fallback_ = std::make_unique<CensusFeatureExtractor>(9);
    }
}

DLFeatureExtractor::~DLFeatureExtractor() = default;

Result<void> DLFeatureExtractor::loadModel() const {
    if (load_attempted_) {
        return loaded_ ? Result<void>::success()
                       : Result<void>::error("DL model initialization has already failed");
    }
    load_attempted_ = true;
    engine_ = std::make_unique<DLInferenceEngine>(config_);
    auto status = engine_->initialize();
    loaded_ = status.has_value();
    return status;
}

Result<FeatureTensor> DLFeatureExtractor::extract(const cv::Mat& image) const {
    auto load_status = loadModel();
    if (!load_status) {
        if (fallback_) {
            return fallback_->extract(image);
        }
        return Result<FeatureTensor>::error(
            "DLFeatureExtractor failed to initialize with model " +
            config_.model_weights_path.string() + ": " + load_status.error_msg());
    }

    auto inference = engine_->infer(image);
    if (!inference) {
        if (fallback_) {
            return fallback_->extract(image);
        }
        return Result<FeatureTensor>::error(inference.error_msg());
    }
    if (inference.value().output.empty()) {
        if (fallback_) {
            return fallback_->extract(image);
        }
        return Result<FeatureTensor>::error("DLFeatureExtractor produced an empty output tensor");
    }

    FeatureTensor tensor;
    tensor.data = inference.value().output;
    tensor.channels = std::max(1, tensor.data.channels());
    tensor.downsample_factor = std::max(1, image.rows / std::max(1, tensor.data.rows));
    feature_channels_ = tensor.channels;
    downsample_factor_ = tensor.downsample_factor;
    return Result<FeatureTensor>::success(tensor);
}

int DLFeatureExtractor::getDownsampleFactor() const {
    if (loaded_) return downsample_factor_;
    return fallback_ ? fallback_->getDownsampleFactor() : 1;
}

int DLFeatureExtractor::getFeatureChannels() const {
    if (loaded_ && feature_channels_ > 0) return feature_channels_;
    return fallback_ ? fallback_->getFeatureChannels() : 0;
}

std::unique_ptr<IFeatureExtractor> createFeatureExtractor(const PipelineConfig& config) {
    if (config.use_dl_features || config.feature_method == FeatureMethod::DL_FEATURE) {
        return std::make_unique<DLFeatureExtractor>(config);
    }
    return std::make_unique<CensusFeatureExtractor>(9);
}

}  // namespace cms
