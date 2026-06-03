#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "config/config_manager.h"

TEST(ConfigManagerTest, LoadsDefaultsAndOverrides) {
    auto tmp = std::filesystem::temp_directory_path() / "cms_test_config.yaml";
    std::ofstream out(tmp);
    out << "max_disparity: 64\n";
    out << "io:\n  manifest_path: ./manifest.csv\n";
    out.close();

    auto result = cms::ConfigManager::load(tmp);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().max_disparity, 64);
    EXPECT_EQ(result.value().sgm_p1, 10);
    EXPECT_EQ(result.value().manifest_path.string(), "./manifest.csv");

    std::filesystem::remove(tmp);
}

TEST(ConfigManagerTest, RejectsInvalidRange) {
    cms::PipelineConfig config;
    config.manifest_path = "manifest.csv";
    config.max_disparity = -1;
    auto result = cms::ConfigManager::validate(config);
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error_msg().find("max_disparity"), std::string::npos);
}

TEST(ConfigManagerTest, RejectsMissingManifestPath) {
    cms::PipelineConfig config;
    auto result = cms::ConfigManager::validate(config);
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error_msg().find("manifest_path"), std::string::npos);
}

TEST(ConfigManagerTest, RejectsMalformedYaml) {
    auto tmp = std::filesystem::temp_directory_path() / "cms_test_bad_config.yaml";
    std::ofstream out(tmp);
    out << "max_disparity: [1, 2\n";
    out.close();

    auto result = cms::ConfigManager::load(tmp);
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error_msg().find("YAML"), std::string::npos);

    std::filesystem::remove(tmp);
}
