#include <cstdlib>
#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include "config/config_manager.h"

RC_GTEST_PROP(ConfigValidationProperty, InvalidMaxDisparityRejected, (int raw_value)) {
    cms::PipelineConfig config;
    config.manifest_path = "manifest.csv";
    config.max_disparity = -(std::abs(raw_value) % 128 + 1);
    auto result = cms::ConfigManager::validate(config);
    RC_ASSERT(!result.has_value());
    RC_ASSERT(result.error_msg().find("max_disparity") != std::string::npos);
}

RC_GTEST_PROP(ConfigValidationProperty, InvalidPenaltyOrderingRejected, (int raw_p1, int raw_p2)) {
    cms::PipelineConfig config;
    config.manifest_path = "manifest.csv";
    config.sgm_p1 = std::abs(raw_p1) % 128;
    config.sgm_p2 = std::abs(raw_p2) % (config.sgm_p1 + 1);
    auto result = cms::ConfigManager::validate(config);
    RC_ASSERT(!result.has_value());
    RC_ASSERT(result.error_msg().find("sgm.p2") != std::string::npos);
}
