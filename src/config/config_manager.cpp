#include "config/config_manager.h"

#include <iostream>
#include <sstream>

#include <yaml-cpp/yaml.h>

namespace cms {
namespace {

template <typename EnumT>
Result<EnumT> invalidEnum(const std::string& field, const std::string& value) {
    return Result<EnumT>::error("Invalid value for " + field + ": " + value);
}

Result<FeatureMethod> parseFeatureMethod(const std::string& value) {
    if (value == "CENSUS") return Result<FeatureMethod>::success(FeatureMethod::CENSUS);
    if (value == "GRADIENT_ORIENTATION") return Result<FeatureMethod>::success(FeatureMethod::GRADIENT_ORIENTATION);
    if (value == "DL_FEATURE") return Result<FeatureMethod>::success(FeatureMethod::DL_FEATURE);
    return invalidEnum<FeatureMethod>("feature_method", value);
}

Result<CostMetric> parseCostMetric(const std::string& value) {
    if (value == "NCC") return Result<CostMetric>::success(CostMetric::NCC);
    if (value == "CENSUS_HAMMING") return Result<CostMetric>::success(CostMetric::CENSUS_HAMMING);
    if (value == "COSINE_SIMILARITY") return Result<CostMetric>::success(CostMetric::COSINE_SIMILARITY);
    return invalidEnum<CostMetric>("cost_metric", value);
}

Result<AggregationMethod> parseAggregationMethod(const std::string& value) {
    if (value == "SGM") return Result<AggregationMethod>::success(AggregationMethod::SGM);
    if (value == "GUIDED_FILTER") return Result<AggregationMethod>::success(AggregationMethod::GUIDED_FILTER);
    return invalidEnum<AggregationMethod>("aggregation_method", value);
}

Result<InferenceBackend> parseInferenceBackend(const std::string& value) {
    if (value == "ONNX_RUNTIME") return Result<InferenceBackend>::success(InferenceBackend::ONNX_RUNTIME);
    if (value == "LIBTORCH") return Result<InferenceBackend>::success(InferenceBackend::LIBTORCH);
    if (value == "OPENCV_DNN") return Result<InferenceBackend>::success(InferenceBackend::OPENCV_DNN);
    if (value == "TENSORRT") return Result<InferenceBackend>::success(InferenceBackend::TENSORRT);
    return invalidEnum<InferenceBackend>("inference_backend", value);
}

const char* toString(FeatureMethod value) {
    switch (value) {
        case FeatureMethod::CENSUS: return "CENSUS";
        case FeatureMethod::GRADIENT_ORIENTATION: return "GRADIENT_ORIENTATION";
        case FeatureMethod::DL_FEATURE: return "DL_FEATURE";
    }
    return "UNKNOWN";
}

const char* toString(CostMetric value) {
    switch (value) {
        case CostMetric::NCC: return "NCC";
        case CostMetric::CENSUS_HAMMING: return "CENSUS_HAMMING";
        case CostMetric::COSINE_SIMILARITY: return "COSINE_SIMILARITY";
    }
    return "UNKNOWN";
}

const char* toString(AggregationMethod value) {
    switch (value) {
        case AggregationMethod::SGM: return "SGM";
        case AggregationMethod::GUIDED_FILTER: return "GUIDED_FILTER";
    }
    return "UNKNOWN";
}

const char* toString(InferenceBackend value) {
    switch (value) {
        case InferenceBackend::ONNX_RUNTIME: return "ONNX_RUNTIME";
        case InferenceBackend::LIBTORCH: return "LIBTORCH";
        case InferenceBackend::OPENCV_DNN: return "OPENCV_DNN";
        case InferenceBackend::TENSORRT: return "TENSORRT";
    }
    return "UNKNOWN";
}

}  // namespace

Result<PipelineConfig> ConfigManager::load(const std::filesystem::path& config_path) {
    if (!std::filesystem::exists(config_path)) {
        return Result<PipelineConfig>::error("Configuration file does not exist: " + config_path.string());
    }

    PipelineConfig config;

    try {
        const YAML::Node root = YAML::LoadFile(config_path.string());

        if (root["max_disparity"]) config.max_disparity = root["max_disparity"].as<int>();
        if (root["feature_method"]) {
            auto parsed = parseFeatureMethod(root["feature_method"].as<std::string>());
            if (!parsed) return Result<PipelineConfig>::error(parsed.error_msg());
            config.feature_method = parsed.value();
        }
        if (root["cost_metric"]) {
            auto parsed = parseCostMetric(root["cost_metric"].as<std::string>());
            if (!parsed) return Result<PipelineConfig>::error(parsed.error_msg());
            config.cost_metric = parsed.value();
        }
        if (root["aggregation_method"]) {
            auto parsed = parseAggregationMethod(root["aggregation_method"].as<std::string>());
            if (!parsed) return Result<PipelineConfig>::error(parsed.error_msg());
            config.aggregation_method = parsed.value();
        }

        if (const YAML::Node sgm = root["sgm"]) {
            if (sgm["num_directions"]) config.sgm_num_directions = sgm["num_directions"].as<int>();
            if (sgm["p1"]) config.sgm_p1 = sgm["p1"].as<int>();
            if (sgm["p2"]) config.sgm_p2 = sgm["p2"].as<int>();
        }

        if (const YAML::Node post = root["post_processing"]) {
            if (post["enable_lr_check"]) config.enable_lr_check = post["enable_lr_check"].as<bool>();
            if (post["enable_hole_filling"]) config.enable_hole_filling = post["enable_hole_filling"].as<bool>();
            if (post["lr_threshold"]) config.lr_threshold = post["lr_threshold"].as<double>();
        }

        if (const YAML::Node depth = root["depth"]) {
            if (depth["depth_max_meter"]) config.depth_max_meter = depth["depth_max_meter"].as<double>();
        }

        if (const YAML::Node prep = root["preprocessing"]) {
            if (prep["enable_clahe"]) config.enable_clahe = prep["enable_clahe"].as<bool>();
            if (prep["clahe_clip_limit"]) config.clahe_clip_limit = prep["clahe_clip_limit"].as<int>();
            if (prep["clahe_tile_size"]) config.clahe_tile_size = prep["clahe_tile_size"].as<int>();
            if (prep["enable_gradient_alignment"]) {
                config.enable_gradient_alignment = prep["enable_gradient_alignment"].as<bool>();
            }
        }

        if (const YAML::Node dl = root["dl"]) {
            if (dl["use_dl_features"]) config.use_dl_features = dl["use_dl_features"].as<bool>();
            if (dl["fallback_to_handcraft"]) config.dl_fallback_to_handcraft = dl["fallback_to_handcraft"].as<bool>();
            if (dl["inference_backend"]) {
                auto parsed = parseInferenceBackend(dl["inference_backend"].as<std::string>());
                if (!parsed) return Result<PipelineConfig>::error(parsed.error_msg());
                config.inference_backend = parsed.value();
            }
            if (dl["model_weights_path"]) config.model_weights_path = dl["model_weights_path"].as<std::string>();
            if (dl["use_gpu"]) config.use_gpu = dl["use_gpu"].as<bool>();
        }

        if (const YAML::Node io = root["io"]) {
            if (io["manifest_path"]) config.manifest_path = io["manifest_path"].as<std::string>();
            if (io["output_dir"]) config.output_dir = io["output_dir"].as<std::string>();
        }
    } catch (const YAML::ParserException& e) {
        return Result<PipelineConfig>::error(std::string("YAML parse error: ") + e.what());
    } catch (const YAML::Exception& e) {
        return Result<PipelineConfig>::error(std::string("YAML error: ") + e.what());
    } catch (const std::exception& e) {
        return Result<PipelineConfig>::error(std::string("Configuration load failed: ") + e.what());
    }

    auto validation = validate(config);
    if (!validation) {
        return Result<PipelineConfig>::error(validation.error_msg());
    }
    return Result<PipelineConfig>::success(config);
}

Result<void> ConfigManager::validate(const PipelineConfig& config) {
    std::ostringstream oss;
    bool valid = true;

    auto addError = [&](const std::string& message) {
        if (!valid) oss << "; ";
        oss << message;
        valid = false;
    };

    if (config.max_disparity <= 0) addError("max_disparity must be > 0");
    if (config.sgm_p1 < 0) addError("sgm.p1 must be >= 0");
    if (config.sgm_p2 <= config.sgm_p1) addError("sgm.p2 must be > sgm.p1");
    if (config.depth_max_meter <= 0.0) addError("depth.depth_max_meter must be > 0");
    if (config.lr_threshold < 0.0) addError("post_processing.lr_threshold must be >= 0");
    if (config.sgm_num_directions != 4 && config.sgm_num_directions != 8 && config.sgm_num_directions != 16) {
        addError("sgm.num_directions must be one of {4, 8, 16}");
    }
    if (config.clahe_clip_limit <= 0) addError("preprocessing.clahe_clip_limit must be > 0");
    if (config.clahe_tile_size <= 0) addError("preprocessing.clahe_tile_size must be > 0");
    if (config.manifest_path.empty()) addError("io.manifest_path is required");

    if (!valid) {
        return Result<void>::error(oss.str());
    }
    return Result<void>::success();
}

void ConfigManager::logSummary(const PipelineConfig& config) {
    std::cout
        << "PipelineConfig summary\n"
        << "  max_disparity: " << config.max_disparity << '\n'
        << "  feature_method: " << toString(config.feature_method) << '\n'
        << "  cost_metric: " << toString(config.cost_metric) << '\n'
        << "  aggregation_method: " << toString(config.aggregation_method) << '\n'
        << "  sgm.num_directions: " << config.sgm_num_directions << '\n'
        << "  sgm.p1: " << config.sgm_p1 << '\n'
        << "  sgm.p2: " << config.sgm_p2 << '\n'
        << "  post_processing.enable_lr_check: " << config.enable_lr_check << '\n'
        << "  post_processing.enable_hole_filling: " << config.enable_hole_filling << '\n'
        << "  post_processing.lr_threshold: " << config.lr_threshold << '\n'
        << "  depth.depth_max_meter: " << config.depth_max_meter << '\n'
        << "  preprocessing.enable_clahe: " << config.enable_clahe << '\n'
        << "  preprocessing.clahe_clip_limit: " << config.clahe_clip_limit << '\n'
        << "  preprocessing.clahe_tile_size: " << config.clahe_tile_size << '\n'
        << "  preprocessing.enable_gradient_alignment: " << config.enable_gradient_alignment << '\n'
        << "  dl.use_dl_features: " << config.use_dl_features << '\n'
        << "  dl.fallback_to_handcraft: " << config.dl_fallback_to_handcraft << '\n'
        << "  dl.inference_backend: " << toString(config.inference_backend) << '\n'
        << "  dl.model_weights_path: " << config.model_weights_path.string() << '\n'
        << "  dl.use_gpu: " << config.use_gpu << '\n'
        << "  io.manifest_path: " << config.manifest_path.string() << '\n'
        << "  io.output_dir: " << config.output_dir.string() << std::endl;
}

}  // namespace cms
