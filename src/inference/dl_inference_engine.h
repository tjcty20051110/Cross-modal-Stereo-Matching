#pragma once

#include <chrono>
#include <memory>

#include <opencv2/core.hpp>

#include "common/types.h"
#include "config/config_manager.h"

namespace cms {

struct InferenceResult {
    cv::Mat output;
    std::chrono::microseconds elapsed_time{0};
};

class DLInferenceEngine {
public:
    explicit DLInferenceEngine(const PipelineConfig& config);
    ~DLInferenceEngine();

    Result<void> initialize();
    Result<InferenceResult> infer(const cv::Mat& input) const;
    bool isUsingGPU() const;

private:
    PipelineConfig config_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool using_gpu_ = false;
};

}  // namespace cms
