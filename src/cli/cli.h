#pragma once

#include <string>
#include <vector>

namespace cms {

class CLI {
public:
    int run(int argc, char* argv[]);

private:
    int cmdDownload(const std::vector<std::string>& args);
    int cmdRun(const std::vector<std::string>& args);
    int cmdEvaluate(const std::vector<std::string>& args);
    int cmdVisualize(const std::vector<std::string>& args);
    int cmdAnalyze(const std::vector<std::string>& args);
    int cmdTrain(const std::vector<std::string>& args);
    int cmdCfmTrain(const std::vector<std::string>& args);
    int cmdCfmInfer(const std::vector<std::string>& args);
    int cmdExportMdpBackbone(const std::vector<std::string>& args);
    int cmdSplitManifest(const std::vector<std::string>& args);

    void printUsage() const;
    void printTiming(const std::string& command, double elapsed_sec, int sample_count) const;
};

}  // namespace cms
