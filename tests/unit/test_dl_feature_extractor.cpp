#include <gtest/gtest.h>

#include "features/feature_extractor.h"

TEST(DLFeatureExtractorTest, FallsBackToCensusWhenModelMissingAndFallbackEnabled) {
    cms::PipelineConfig config;
    config.use_dl_features = true;
    config.feature_method = cms::FeatureMethod::DL_FEATURE;
    config.dl_fallback_to_handcraft = true;
    config.inference_backend = cms::InferenceBackend::OPENCV_DNN;
    config.model_weights_path = "/tmp/nonexistent_model.onnx";
    config.manifest_path = "manifest.csv";

    cms::DLFeatureExtractor extractor(config);
    cv::Mat image(8, 8, CV_32F, cv::Scalar(0.5f));
    auto result = extractor.extract(image);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().channels, 80);
}

TEST(DLFeatureExtractorTest, ReturnsErrorWhenModelMissingAndFallbackDisabled) {
    cms::PipelineConfig config;
    config.use_dl_features = true;
    config.feature_method = cms::FeatureMethod::DL_FEATURE;
    config.dl_fallback_to_handcraft = false;
    config.inference_backend = cms::InferenceBackend::OPENCV_DNN;
    config.model_weights_path = "/tmp/nonexistent_model.onnx";
    config.manifest_path = "manifest.csv";

    cms::DLFeatureExtractor extractor(config);
    cv::Mat image(8, 8, CV_32F, cv::Scalar(0.5f));
    auto result = extractor.extract(image);
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error_msg().find("nonexistent_model.onnx"), std::string::npos);
}
