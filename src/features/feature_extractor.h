#pragma once

#include <memory>

#include <opencv2/core.hpp>

#include "common/types.h"
#include "config/config_manager.h"

namespace cms {

class DLInferenceEngine;

struct FeatureTensor {
    cv::Mat data;
    int channels = 0;
    int downsample_factor = 1;
};

class IFeatureExtractor {
public:
    virtual ~IFeatureExtractor() = default;
    virtual Result<FeatureTensor> extract(const cv::Mat& image) const = 0;
    virtual int getDownsampleFactor() const = 0;
    virtual int getFeatureChannels() const = 0;
};

class CensusFeatureExtractor : public IFeatureExtractor {
public:
    explicit CensusFeatureExtractor(int window_size = 9);
    Result<FeatureTensor> extract(const cv::Mat& image) const override;
    int getDownsampleFactor() const override { return 1; }
    int getFeatureChannels() const override;

private:
    int window_size_;
};

class DLFeatureExtractor : public IFeatureExtractor {
public:
    explicit DLFeatureExtractor(const PipelineConfig& config);
    ~DLFeatureExtractor() override;
    Result<FeatureTensor> extract(const cv::Mat& image) const override;
    int getDownsampleFactor() const override;
    int getFeatureChannels() const override;
    Result<void> loadModel() const;

private:
    PipelineConfig config_;
    mutable std::unique_ptr<DLInferenceEngine> engine_;
    mutable std::unique_ptr<IFeatureExtractor> fallback_;
    mutable bool load_attempted_ = false;
    mutable bool loaded_ = false;
    mutable int feature_channels_ = 0;
    mutable int downsample_factor_ = 1;
};

std::unique_ptr<IFeatureExtractor> createFeatureExtractor(const PipelineConfig& config);

}  // namespace cms
