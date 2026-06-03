#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

namespace {

std::filesystem::path repoRoot() {
    return std::filesystem::path(CMS_SOURCE_DIR);
}

std::filesystem::path makeTempDir(const std::string& prefix) {
    const auto unique = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    auto dir = std::filesystem::temp_directory_path() / (prefix + "_" + unique);
    std::filesystem::create_directories(dir);
    return dir;
}

int runPythonHelper(const std::filesystem::path& working_dir,
                    const std::string& script_body,
                    const std::vector<std::string>& args = {}) {
    const auto helper = working_dir / "helper.py";
    std::ofstream out(helper);
    out << script_body;
    out.close();

    std::ostringstream command;
    command << "python3 \"" << helper.string() << "\"";
    for (const auto& arg : args) {
        command << " \"" << arg << "\"";
    }
    return std::system(command.str().c_str());
}

std::vector<std::vector<std::string>> readCsv(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty() && cell.back() == '\r') {
                cell.pop_back();
            }
            row.push_back(cell);
        }
        rows.push_back(row);
    }
    return rows;
}

}  // namespace

RC_GTEST_PROP(ManifestGenerationProperty, ManifestContainsOneRowPerStereoPair, (int raw_count, bool use_nir)) {
    const int pair_count = std::abs(raw_count) % 5 + 1;
    const auto root = makeTempDir("cms_manifest_prop");
    const auto split_dir = root / "MS2" / "train" / "seq01";
    const auto rgb_dir = split_dir / "RGB";
    const auto ir_dir = split_dir / (use_nir ? "NIR" : "ThermalIR");
    const auto gt_dir = split_dir / "disparity";
    const auto calib_dir = split_dir / "calib";
    std::filesystem::create_directories(rgb_dir);
    std::filesystem::create_directories(ir_dir);
    std::filesystem::create_directories(gt_dir);
    std::filesystem::create_directories(calib_dir);

    std::ofstream calib(calib_dir / "camera.yaml");
    calib << "focal_length: 320.0\nbaseline: 0.2\n";
    calib.close();

    for (int i = 0; i < pair_count; ++i) {
        const auto stem = "frame_" + std::to_string(i);
        std::ofstream(rgb_dir / (stem + ".png")) << "rgb";
        std::ofstream(ir_dir / (stem + ".png")) << "ir";
        std::ofstream(gt_dir / (stem + ".png")) << "gt";
    }

    const auto manifest = root / "manifest.csv";
    std::ostringstream helper;
    helper << "import importlib.util\n";
    helper << "from pathlib import Path\n";
    helper << "import sys\n";
    helper << "spec = importlib.util.spec_from_file_location('download_ms2', r'" << (repoRoot() / "scripts" / "download_ms2.py").string() << "')\n";
    helper << "mod = importlib.util.module_from_spec(spec)\n";
    helper << "sys.modules[spec.name] = mod\n";
    helper << "spec.loader.exec_module(mod)\n";
    helper << "mod.generate_manifest(Path(r'" << root.string() << "'), ['train'], Path(r'" << manifest.string() << "'))\n";

    RC_ASSERT(runPythonHelper(root, helper.str()) == 0);
    RC_ASSERT(std::filesystem::exists(manifest));

    const auto rows = readCsv(manifest);
    RC_ASSERT(!rows.empty());
    RC_ASSERT(rows.front().size() == 5);
    RC_ASSERT(static_cast<int>(rows.size()) == static_cast<int>(pair_count + 1));

    for (size_t i = 1; i < rows.size(); ++i) {
        RC_ASSERT(rows[i].size() == 5);
        RC_ASSERT(!rows[i][0].empty());
        RC_ASSERT(!rows[i][1].empty());
        RC_ASSERT(!rows[i][2].empty());
        RC_ASSERT(!rows[i][3].empty());
        RC_ASSERT(rows[i][4] == "train");
    }

    std::filesystem::remove_all(root);
}

RC_GTEST_PROP(ManifestGenerationProperty, ShaMismatchRejectsAndLeavesNoFile, (int raw_size)) {
    const int payload_size = std::abs(raw_size) % 128 + 1;
    const auto root = makeTempDir("cms_sha_prop");
    std::ostringstream helper;
    helper << "import importlib.util\n";
    helper << "from pathlib import Path\n";
    helper << "import sys\n";
    helper << "spec = importlib.util.spec_from_file_location('download_ms2', r'" << (repoRoot() / "scripts" / "download_ms2.py").string() << "')\n";
    helper << "mod = importlib.util.module_from_spec(spec)\n";
    helper << "sys.modules[spec.name] = mod\n";
    helper << "spec.loader.exec_module(mod)\n";
    helper << "class FakeResponse:\n";
    helper << "    status_code = 200\n";
    helper << "    headers = {'Content-Length': '" << payload_size << "'}\n";
    helper << "    def iter_content(self, chunk_size=8192):\n";
    helper << "        yield (b'x' * " << payload_size << ")\n";
    helper << "class FakeSession:\n";
    helper << "    def get(self, url, headers=None, stream=True, timeout=60):\n";
    helper << "        return FakeResponse()\n";
    helper << "dest = Path(r'" << (root / "bad.bin").string() << "')\n";
    helper << "ok = mod.download_file('http://example.invalid/file.bin', dest, '0' * 64, session=FakeSession())\n";
    helper << "tmp = dest.with_suffix(dest.suffix + '.tmp')\n";
    helper << "raise SystemExit(0 if (not ok and not dest.exists() and not tmp.exists()) else 1)\n";

    RC_ASSERT(runPythonHelper(root, helper.str()) == 0);
    RC_ASSERT(!std::filesystem::exists(root / "bad.bin"));
    RC_ASSERT(!std::filesystem::exists(root / "bad.bin.tmp"));

    std::filesystem::remove_all(root);
}
