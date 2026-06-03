// cfm_checkpoint_meta.cpp — Implementation of CFMCheckpointMeta read/write.
#include "cfm/cfm_checkpoint_meta.h"

#ifdef CMS_HAS_LIBTORCH

#include <fstream>
#include <iostream>
#include <sstream>

namespace cms {
namespace cfm {

std::filesystem::path CFMCheckpointMeta::pathFor(
    const std::filesystem::path& model_path) {
    auto p = model_path;
    p += ".meta.json";
    return p;
}

CFMCheckpointMeta CFMCheckpointMeta::legacyDefaults() {
    CFMCheckpointMeta m;
    m.schema_version = 1;
    m.attention_mode = "legacy_gated";
    m.mdp_backbone = "lightweight_cnn";
    m.num_heads = 4;
    m.feature_dim = 32;
    m.num_disparities = 48;
    m.depth_min = 1.0;
    m.depth_max = 80.0;
    return m;
}

// Minimal JSON writer (avoids external dependency).
namespace {

std::string escapeJson(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    return out;
}

// Trim whitespace from both ends.
std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Extract a JSON string value for a given key from a flat JSON object.
// This is a minimal parser sufficient for our single-level schema.
std::string extractString(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    ++pos;
    auto end = json.find('"', pos);
    if (end == std::string::npos) return "";
    return json.substr(pos, end - pos);
}

// Extract a numeric value (int or double) as string.
std::string extractNumber(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    std::string num;
    while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '.' ||
                                  json[pos] == '-' || json[pos] == 'e' || json[pos] == 'E' || json[pos] == '+')) {
        num += json[pos++];
    }
    return num;
}

}  // namespace

Result<void> CFMCheckpointMeta::write(const std::filesystem::path& meta_path) const {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"schema_version\": " << schema_version << ",\n";
    oss << "  \"attention_mode\": \"" << escapeJson(attention_mode) << "\",\n";
    oss << "  \"mdp_backbone\": \"" << escapeJson(mdp_backbone) << "\",\n";
    oss << "  \"mdp_backbone_weights_source\": \"" << escapeJson(mdp_backbone_weights_source) << "\",\n";
    oss << "  \"num_disparities\": " << num_disparities << ",\n";
    oss << std::fixed;
    oss.precision(6);
    oss << "  \"depth_min\": " << depth_min << ",\n";
    oss << "  \"depth_max\": " << depth_max << ",\n";
    oss << "  \"feature_dim\": " << feature_dim << ",\n";
    oss << "  \"num_heads\": " << num_heads << ",\n";
    oss << "  \"git_sha\": \"" << escapeJson(git_sha) << "\",\n";
    oss << "  \"training_config_hash\": \"" << escapeJson(training_config_hash) << "\"\n";
    oss << "}\n";

    auto tmp_path = meta_path;
    tmp_path += ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        if (!f) {
            return Result<void>::error("failed to open for writing: " + tmp_path.string());
        }
        f << oss.str();
        if (!f) {
            return Result<void>::error("failed while writing: " + tmp_path.string());
        }
    }

    std::error_code ec;
    std::filesystem::rename(tmp_path, meta_path, ec);
    if (ec) {
        return Result<void>::error("failed to rename " + tmp_path.string() +
                                   " -> " + meta_path.string() + ": " + ec.message());
    }
    return Result<void>::success();
}

Result<CFMCheckpointMeta> CFMCheckpointMeta::read(
    const std::filesystem::path& meta_path) {
    if (!std::filesystem::exists(meta_path)) {
        return Result<CFMCheckpointMeta>::error("missing");
    }

    std::ifstream f(meta_path, std::ios::binary);
    if (!f) {
        return Result<CFMCheckpointMeta>::error(
            "failed to open metadata file: " + meta_path.string());
    }
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    // Basic validation: must contain opening brace.
    if (content.find('{') == std::string::npos) {
        return Result<CFMCheckpointMeta>::error(
            "malformed JSON (no opening brace): " + meta_path.string());
    }

    CFMCheckpointMeta meta;
    auto sv = extractNumber(content, "schema_version");
    if (!sv.empty()) meta.schema_version = std::stoi(sv);

    meta.attention_mode = extractString(content, "attention_mode");
    meta.mdp_backbone = extractString(content, "mdp_backbone");
    meta.mdp_backbone_weights_source = extractString(content, "mdp_backbone_weights_source");

    auto nd = extractNumber(content, "num_disparities");
    if (!nd.empty()) meta.num_disparities = std::stoi(nd);

    auto dmin = extractNumber(content, "depth_min");
    if (!dmin.empty()) meta.depth_min = std::stod(dmin);

    auto dmax = extractNumber(content, "depth_max");
    if (!dmax.empty()) meta.depth_max = std::stod(dmax);

    auto fd = extractNumber(content, "feature_dim");
    if (!fd.empty()) meta.feature_dim = std::stoi(fd);

    auto nh = extractNumber(content, "num_heads");
    if (!nh.empty()) meta.num_heads = std::stoi(nh);

    meta.git_sha = extractString(content, "git_sha");
    meta.training_config_hash = extractString(content, "training_config_hash");

    // Validate required fields.
    if (meta.attention_mode.empty()) {
        return Result<CFMCheckpointMeta>::error(
            "malformed metadata: missing 'attention_mode' in " + meta_path.string());
    }
    if (meta.mdp_backbone.empty()) {
        return Result<CFMCheckpointMeta>::error(
            "malformed metadata: missing 'mdp_backbone' in " + meta_path.string());
    }

    return Result<CFMCheckpointMeta>::success(meta);
}

}  // namespace cfm
}  // namespace cms

#endif  // CMS_HAS_LIBTORCH
