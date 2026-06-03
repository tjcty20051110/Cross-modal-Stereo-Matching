#include "cli/cli.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "config/config_manager.h"
#include "data/data_loader.h"
#include "data/manifest_splitter.h"
#include "evaluation/evaluator.h"
#include "preprocessing/preprocessor.h"
#include "stereo/stereo_matcher.h"
#include "visualization/visualizer.h"

#ifdef CMS_HAS_LIBTORCH
#include "training/trainer.h"
#include "cfm/cfm_trainer.h"
#include "cfm/cfm_inference.h"
#include "cfm/cfm_attention.h"
#include "cfm/mdp_backbone.h"
#include "cfm/backbone_exporter.h"
#endif

namespace cms {
namespace {
std::string getOption(const std::vector<std::string>& args,
                      const std::string& key,
                      const std::string& default_value = "") {
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == key) return args[i + 1];
    }
    return default_value;
}

std::vector<std::string> getPositionalArgs(const std::vector<std::string>& args) {
    std::vector<std::string> positional;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i].rfind("--", 0) == 0) {
            if (i + 1 < args.size() && args[i + 1].rfind("--", 0) != 0) {
                ++i;
            }
            continue;
        }
        positional.push_back(args[i]);
    }
    return positional;
}

cv::Mat loadPrediction(const std::filesystem::path& pred_dir, const std::string& sample_id) {
    cv::Mat pred16 = cv::imread((pred_dir / (sample_id + "_disp.png")).string(), cv::IMREAD_UNCHANGED);
    cv::Mat pred32;
    if (!pred16.empty()) pred16.convertTo(pred32, CV_32F, 1.0 / 256.0);
    return pred32;
}

cv::Mat loadMaskOrDefault(const std::filesystem::path& pred_dir,
                         const std::string& sample_id,
                         const cv::Mat& fallback) {
    cv::Mat mask = cv::imread((pred_dir / (sample_id + "_mask.png")).string(), cv::IMREAD_GRAYSCALE);
    return mask.empty() ? fallback : mask;
}

struct AnalysisStats {
    double left_mean_sum = 0.0;
    double right_mean_sum = 0.0;
    double left_grad_sum = 0.0;
    double right_grad_sum = 0.0;
    int used = 0;
};

cv::Mat makeSummaryCanvas(const AnalysisStats& stats,
                          double census_avg_epe,
                          double ncc_avg_epe) {
    cv::Mat summary(520, 900, CV_8UC3, cv::Scalar(32, 32, 32));
    auto drawBar = [&](int x, int y, const std::string& label, double value,
                       const cv::Scalar& color, double scale) {
        cv::putText(summary, label, cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        int width = static_cast<int>(std::max(0.0, std::min(scale, value)) * 280.0 / scale);
        cv::rectangle(summary, cv::Rect(x, y, width, 30), color, cv::FILLED);
        cv::rectangle(summary, cv::Rect(x, y, 280, 30), cv::Scalar(200, 200, 200), 1);
        cv::putText(summary, cv::format("%.4f", value), cv::Point(x + 290, y + 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    };

    cv::putText(summary, "Cross-modal analysis summary", cv::Point(50, 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.95, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    if (stats.used > 0) {
        drawBar(50, 90, "Left mean", stats.left_mean_sum / stats.used,
                cv::Scalar(0, 180, 255), 1.0);
        drawBar(50, 160, "Right mean", stats.right_mean_sum / stats.used,
                cv::Scalar(0, 255, 180), 1.0);
        drawBar(50, 230, "Left grad", stats.left_grad_sum / stats.used,
                cv::Scalar(255, 180, 0), 1.0);
        drawBar(50, 300, "Right grad", stats.right_grad_sum / stats.used,
                cv::Scalar(255, 0, 180), 1.0);
    }

    drawBar(50, 390, "Census EPE",
            std::isfinite(census_avg_epe) ? census_avg_epe : 0.0,
            cv::Scalar(30, 200, 255), 10.0);
    drawBar(50, 460, "NCC EPE",
            std::isfinite(ncc_avg_epe) ? ncc_avg_epe : 0.0,
            cv::Scalar(120, 255, 80), 10.0);
    return summary;
}

}  // namespace

int CLI::run(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    const std::string command = argv[1];
    std::vector<std::string> args;
    for (int i = 2; i < argc; ++i) args.emplace_back(argv[i]);

    if (command == "download") return cmdDownload(args);
    if (command == "run") return cmdRun(args);
    if (command == "evaluate") return cmdEvaluate(args);
    if (command == "visualize") return cmdVisualize(args);
    if (command == "analyze") return cmdAnalyze(args);
    if (command == "train") return cmdTrain(args);
    if (command == "cfm-train") return cmdCfmTrain(args);
    if (command == "cfm-infer") return cmdCfmInfer(args);
    if (command == "export-mdp-backbone") return cmdExportMdpBackbone(args);
    if (command == "split-manifest") return cmdSplitManifest(args);

    printUsage();
    return 1;
}

int CLI::cmdDownload(const std::vector<std::string>& args) {
    const auto start = std::chrono::steady_clock::now();
    std::filesystem::path script = std::filesystem::current_path() / "scripts" / "download_ms2.py";
    std::string command = "python3 \"" + script.string() + "\"";
    for (const auto& arg : args) command += " \"" + arg + "\"";
    int rc = std::system(command.c_str());
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printTiming("download", elapsed, 0);
    return rc;
}

int CLI::cmdRun(const std::vector<std::string>& args) {
    const auto start = std::chrono::steady_clock::now();
    auto config_path = std::filesystem::path(getOption(args, "--config", "configs/default.yaml"));
    auto config_res = ConfigManager::load(config_path);
    if (!config_res) {
        std::cerr << config_res.error_msg() << std::endl;
        return 1;
    }
    auto config = config_res.value();
    std::string manifest_override = getOption(args, "--manifest");
    if (!manifest_override.empty()) config.manifest_path = manifest_override;
    std::string output_override = getOption(args, "--output-dir");
    if (!output_override.empty()) config.output_dir = output_override;

    auto valid = ConfigManager::validate(config);
    if (!valid) {
        std::cerr << valid.error_msg() << std::endl;
        return 1;
    }
    ConfigManager::logSummary(config);

    DataLoader loader;
    auto count = loader.loadManifest(config.manifest_path);
    if (!count) {
        std::cerr << count.error_msg() << std::endl;
        return 1;
    }

    StereoMatcher matcher(config);
    const auto out_dir = std::filesystem::path(config.output_dir);
    std::filesystem::create_directories(out_dir);

    int success = 0;
    int failed = 0;
    for (size_t i = 0; i < loader.size(); ++i) {
        auto sample = loader.getSample(i);
        if (!sample) {
            ++failed;
            std::cerr << "[sample-load] " << sample.error_msg() << std::endl;
            continue;
        }
        auto result = matcher.match(sample.value());
        if (!result) {
            ++failed;
            std::cerr << "[sample-run] " << sample.value().sample_id << ": " << result.error_msg() << std::endl;
            continue;
        }

        cv::Mat disp16;
        result.value().disparity.convertTo(disp16, CV_16U, 256.0);
        cv::imwrite((out_dir / (sample.value().sample_id + "_disp.png")).string(), disp16);
        cv::imwrite((out_dir / (sample.value().sample_id + "_mask.png")).string(), result.value().confidence_mask);
        ++success;
    }

    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "run finished: success=" << success << ", failed=" << failed << std::endl;
    printTiming("run", elapsed, static_cast<int>(loader.size()));
    return failed == 0 ? 0 : 1;
}

int CLI::cmdEvaluate(const std::vector<std::string>& args) {
    const auto start = std::chrono::steady_clock::now();
    auto config_res = ConfigManager::load(getOption(args, "--config", "configs/default.yaml"));
    if (!config_res) {
        std::cerr << config_res.error_msg() << std::endl;
        return 1;
    }
    auto config = config_res.value();
    std::string manifest_override = getOption(args, "--manifest");
    if (!manifest_override.empty()) config.manifest_path = manifest_override;
    const auto pred_dir = std::filesystem::path(getOption(args, "--pred-dir", config.output_dir.string()));
    const auto csv_path = std::filesystem::path(getOption(args, "--csv", (pred_dir / "metrics.csv").string()));

    DataLoader loader;
    auto count = loader.loadManifest(config.manifest_path);
    if (!count) {
        std::cerr << count.error_msg() << std::endl;
        return 1;
    }

    std::vector<EvalMetrics> metrics;
    for (size_t i = 0; i < loader.size(); ++i) {
        auto sample = loader.getSample(i);
        if (!sample) continue;
        if (sample.value().gt_disparity.empty()) continue;
        cv::Mat pred = loadPrediction(pred_dir, sample.value().sample_id);
        if (pred.empty()) continue;
        cv::Mat mask = loadMaskOrDefault(pred_dir, sample.value().sample_id, sample.value().valid_mask);
        metrics.push_back(Evaluator::evaluate(pred, sample.value().gt_disparity, mask, sample.value().sample_id));
    }

    auto split = Evaluator::summarize(metrics);
    Evaluator::writeCSV(split, csv_path);
    std::cout << "MSE=" << split.average.mse << ", EPE=" << split.average.epe
              << ", D1-all=" << split.average.d1_all << std::endl;
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printTiming("evaluate", elapsed, static_cast<int>(metrics.size()));
    return 0;
}

int CLI::cmdVisualize(const std::vector<std::string>& args) {
    const auto start = std::chrono::steady_clock::now();
    auto config_res = ConfigManager::load(getOption(args, "--config", "configs/default.yaml"));
    if (!config_res) {
        std::cerr << config_res.error_msg() << std::endl;
        return 1;
    }
    auto config = config_res.value();
    std::string manifest_override = getOption(args, "--manifest");
    if (!manifest_override.empty()) config.manifest_path = manifest_override;
    const auto pred_dir = std::filesystem::path(getOption(args, "--pred-dir", config.output_dir.string()));
    const auto viz_dir = std::filesystem::path(getOption(args, "--output-dir", (std::filesystem::path(config.output_dir) / "visualizations").string()));

    std::set<std::string> selected_ids;
    for (const auto& arg : getPositionalArgs(args)) {
        selected_ids.insert(arg);
    }

    DataLoader loader;
    auto count = loader.loadManifest(config.manifest_path);
    if (!count) {
        std::cerr << count.error_msg() << std::endl;
        return 1;
    }

    Visualizer visualizer(config);
    int written = 0;
    for (size_t i = 0; i < loader.size(); ++i) {
        auto sample = loader.getSample(i);
        if (!sample) continue;
        if (!selected_ids.empty() && !selected_ids.count(sample.value().sample_id)) continue;
        cv::Mat pred = loadPrediction(pred_dir, sample.value().sample_id);
        if (pred.empty()) continue;
        auto pred_color = visualizer.renderDisparityColor(pred, config.max_disparity);
        visualizer.save(pred_color, sample.value().sample_id, "pred", viz_dir);
        if (!sample.value().gt_disparity.empty()) {
            auto gt_color = visualizer.renderDisparityColor(sample.value().gt_disparity, config.max_disparity);
            auto heat = visualizer.renderErrorHeatmap(pred, sample.value().gt_disparity, sample.value().valid_mask);
            auto comp = visualizer.renderComparison(sample.value().left_image, sample.value().right_image,
                                                     pred_color, gt_color, heat, true);
            visualizer.save(gt_color, sample.value().sample_id, "gt", viz_dir);
            visualizer.save(heat, sample.value().sample_id, "error", viz_dir);
            visualizer.save(comp, sample.value().sample_id, "comparison", viz_dir);
        } else {
            auto empty_gt = cv::Mat(pred.size(), CV_8UC3, cv::Scalar(127, 127, 127));
            auto comp = visualizer.renderComparison(sample.value().left_image, sample.value().right_image,
                                                     pred_color, empty_gt, false);
            visualizer.save(comp, sample.value().sample_id, "comparison", viz_dir);
        }
        ++written;
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printTiming("visualize", elapsed, written);
    return 0;
}

int CLI::cmdAnalyze(const std::vector<std::string>& args) {
    const auto start = std::chrono::steady_clock::now();
    auto config_res = ConfigManager::load(getOption(args, "--config", "configs/default.yaml"));
    if (!config_res) {
        std::cerr << config_res.error_msg() << std::endl;
        return 1;
    }
    auto config = config_res.value();
    std::string manifest_override = getOption(args, "--manifest");
    if (!manifest_override.empty()) config.manifest_path = manifest_override;
    const auto out_dir = std::filesystem::path(getOption(args, "--output-dir", (std::filesystem::path(config.output_dir) / "analysis").string()));
    std::filesystem::create_directories(out_dir);

    DataLoader loader;
    auto count = loader.loadManifest(config.manifest_path);
    if (!count) {
        std::cerr << count.error_msg() << std::endl;
        return 1;
    }

    Preprocessor preprocessor(config);
    std::ofstream csv(out_dir / "modality_stats.csv");
    csv << "sample_id,left_mean,right_mean,left_std,right_std,left_grad_mean,right_grad_mean\n";

    AnalysisStats stats;
    std::vector<StereoSample> samples_for_comparison;
    for (size_t i = 0; i < loader.size(); ++i) {
        auto sample = loader.getSample(i);
        if (!sample) continue;
        auto prep = preprocessor.process(sample.value().left_image, sample.value().right_image,
                                         sample.value().left_modality, sample.value().right_modality);
        if (!prep) continue;
        cv::Scalar lmean, lstd, rmean, rstd;
        cv::meanStdDev(prep.value().left, lmean, lstd);
        cv::meanStdDev(prep.value().right, rmean, rstd);
        cv::Scalar lgmean = cv::mean(prep.value().left_gradient.empty() ? prep.value().left : prep.value().left_gradient);
        cv::Scalar rgmean = cv::mean(prep.value().right_gradient.empty() ? prep.value().right : prep.value().right_gradient);
        csv << sample.value().sample_id << ',' << lmean[0] << ',' << rmean[0] << ','
            << lstd[0] << ',' << rstd[0] << ',' << lgmean[0] << ',' << rgmean[0] << '\n';
        stats.left_mean_sum += lmean[0];
        stats.right_mean_sum += rmean[0];
        stats.left_grad_sum += lgmean[0];
        stats.right_grad_sum += rgmean[0];
        ++stats.used;
        if (!sample.value().gt_disparity.empty() && static_cast<int>(samples_for_comparison.size()) < 50) {
            samples_for_comparison.push_back(sample.value());
        }
        if (stats.used >= 50) break;
    }

    PipelineConfig census_config = config;
    census_config.cost_metric = CostMetric::CENSUS_HAMMING;
    PipelineConfig ncc_config = config;
    ncc_config.cost_metric = CostMetric::NCC;

    StereoMatcher census_matcher(census_config);
    StereoMatcher ncc_matcher(ncc_config);
    std::ofstream comparison_csv(out_dir / "cost_metric_comparison.csv");
    comparison_csv << "sample_id,census_epe,ncc_epe,better_metric\n";

    double census_epe_sum = 0.0;
    double ncc_epe_sum = 0.0;
    int compared = 0;
    for (const auto& sample : samples_for_comparison) {
        auto census_result = census_matcher.match(sample);
        auto ncc_result = ncc_matcher.match(sample);
        if (!census_result || !ncc_result) continue;

        auto census_metric = Evaluator::evaluate(
            census_result.value().disparity, sample.gt_disparity, sample.valid_mask, sample.sample_id);
        auto ncc_metric = Evaluator::evaluate(
            ncc_result.value().disparity, sample.gt_disparity, sample.valid_mask, sample.sample_id);
        if (!census_metric.is_valid || !ncc_metric.is_valid) continue;

        const std::string better = census_metric.epe <= ncc_metric.epe ? "CENSUS_HAMMING" : "NCC";
        comparison_csv << sample.sample_id << ',' << census_metric.epe << ',' << ncc_metric.epe << ','
                       << better << '\n';
        census_epe_sum += census_metric.epe;
        ncc_epe_sum += ncc_metric.epe;
        ++compared;
    }

    const double census_avg_epe = compared > 0 ? census_epe_sum / compared : std::numeric_limits<double>::quiet_NaN();
    const double ncc_avg_epe = compared > 0 ? ncc_epe_sum / compared : std::numeric_limits<double>::quiet_NaN();

    cv::Mat summary = makeSummaryCanvas(stats, census_avg_epe, ncc_avg_epe);
    cv::imwrite((out_dir / "modality_summary.png").string(), summary);

    std::ofstream report(out_dir / "modality_analysis_report.md");
    report << "# 模态差异分析报告\n\n";
    report << "## 当前统计范围\n\n";
    report << "- 亮度/梯度统计样本数: " << stats.used << "\n";
    report << "- 代价度量对比样本数: " << compared << "\n\n";
    report << "## 亮度与梯度统计\n\n";
    if (stats.used > 0) {
        report << "- 左图平均亮度: " << (stats.left_mean_sum / stats.used) << "\n";
        report << "- 右图平均亮度: " << (stats.right_mean_sum / stats.used) << "\n";
        report << "- 左图平均梯度: " << (stats.left_grad_sum / stats.used) << "\n";
        report << "- 右图平均梯度: " << (stats.right_grad_sum / stats.used) << "\n\n";
    }
    report << "## 代价度量对比\n\n";
    report << "- Census Hamming 平均 EPE: " << census_avg_epe << "\n";
    report << "- NCC 平均 EPE: " << ncc_avg_epe << "\n";
    if (std::isfinite(census_avg_epe) && std::isfinite(ncc_avg_epe)) {
        report << "- 当前更优方法: "
               << (census_avg_epe <= ncc_avg_epe ? "CENSUS_HAMMING" : "NCC") << "\n";
    }
    report << "\n## 产物\n\n";
    report << "- `modality_stats.csv`\n";
    report << "- `cost_metric_comparison.csv`\n";
    report << "- `modality_summary.png`\n";
    report << "- `modality_analysis_report.md`\n";

    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printTiming("analyze", elapsed, stats.used);
    return 0;
}

int CLI::cmdTrain(const std::vector<std::string>& args) {
    const auto start = std::chrono::steady_clock::now();
#ifndef CMS_HAS_LIBTORCH
    (void)args;
    std::cerr << "train subcommand requires LibTorch. Rebuild with -DCMS_ENABLE_LIBTORCH=ON" << std::endl;
    return 1;
#else
    TrainingConfig train_cfg;
    train_cfg.manifest_path = getOption(args, "--manifest", "data/MS2/manifest.csv");
    train_cfg.output_model_path = getOption(args, "--output", "models/siamese.pt");
    train_cfg.num_epochs = std::stoi(getOption(args, "--epochs", "5"));
    train_cfg.batch_size = std::stoi(getOption(args, "--batch-size", "2"));
    train_cfg.samples_per_image = std::stoi(getOption(args, "--samples-per-image", "512"));
    train_cfg.learning_rate = std::stod(getOption(args, "--lr", "0.001"));
    train_cfg.margin = std::stod(getOption(args, "--margin", "0.2"));
    train_cfg.max_samples = std::stoi(getOption(args, "--max-samples", "0"));
    train_cfg.log_every = std::stoi(getOption(args, "--log-every", "10"));
    const std::string use_gpu = getOption(args, "--use-gpu", "true");
    train_cfg.use_gpu = (use_gpu == "true" || use_gpu == "1");

    std::cout << "[train] manifest=" << train_cfg.manifest_path
              << " output=" << train_cfg.output_model_path
              << " epochs=" << train_cfg.num_epochs
              << " batch_size=" << train_cfg.batch_size
              << " samples_per_image=" << train_cfg.samples_per_image
              << " lr=" << train_cfg.learning_rate
              << " margin=" << train_cfg.margin
              << " max_samples=" << train_cfg.max_samples
              << " use_gpu=" << train_cfg.use_gpu << std::endl;

    auto result = Trainer::train(train_cfg);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printTiming("train", elapsed, 0);
    if (!result) {
        std::cerr << "[train] failed: " << result.error_msg() << std::endl;
        return 1;
    }
    return 0;
#endif
}

int CLI::cmdCfmTrain(const std::vector<std::string>& args) {
    const auto start = std::chrono::steady_clock::now();
#ifndef CMS_HAS_LIBTORCH
    (void)args;
    std::cerr << "cfm-train subcommand requires LibTorch. Rebuild with -DCMS_ENABLE_LIBTORCH=ON" << std::endl;
    return 1;
#else
    cfm::CFMTrainingConfig cfg;
    cfg.manifest_path       = getOption(args, "--manifest", "data/MS2/manifest_nir_rgb_train.csv");
    cfg.output_model_path   = getOption(args, "--output", "models/cfm.pt");
    const std::string stage_s = getOption(args, "--stage", "all");
    if      (stage_s == "cfm")   cfg.stage = cfm::TrainStage::CFM;
    else if (stage_s == "mdp")   cfg.stage = cfm::TrainStage::MDP;
    else if (stage_s == "depth") cfg.stage = cfm::TrainStage::DEPTH;
    else                         cfg.stage = cfm::TrainStage::ALL;

    cfg.num_epochs       = std::stoi(getOption(args, "--epochs", "5"));
    cfg.batch_size       = std::stoi(getOption(args, "--batch-size", "4"));
    cfg.learning_rate    = std::stod(getOption(args, "--lr", "0.0001"));
    cfg.num_disparities  = std::stoi(getOption(args, "--num-disparities", "96"));
    cfg.depth_min        = std::stod(getOption(args, "--depth-min", "1.0"));
    cfg.depth_max        = std::stod(getOption(args, "--depth-max", "80.0"));
    cfg.log_every        = std::stoi(getOption(args, "--log-every", "10"));
    const std::string use_gpu = getOption(args, "--use-gpu", "true");
    cfg.use_gpu          = (use_gpu == "true" || use_gpu == "1");
    cfg.init_from        = getOption(args, "--init-from", "");

    // --max-samples validation (Req 1.2, 1.3).
    const std::string max_s = getOption(args, "--max-samples", "0");
    try {
        cfg.max_samples = std::stoi(max_s);
    } catch (...) {
        std::cerr << "--max-samples must be an integer in [0, 1000000], got: " << max_s << std::endl;
        return 2;
    }
    if (cfg.max_samples < 0 || cfg.max_samples > 1000000) {
        std::cerr << "--max-samples out of range [0, 1000000], got: " << cfg.max_samples << std::endl;
        return 2;
    }

    // --attention (Req 3.7, 5.1).
    const std::string attn_s = getOption(args, "--attention", "real_scaled_dot");
    auto attn_opt = cfm::attentionModeFromString(attn_s);
    if (!attn_opt) {
        std::cerr << "--attention must be one of {legacy_gated, real_scaled_dot}, got: " << attn_s << std::endl;
        return 2;
    }
    cfg.attention_mode = *attn_opt;

    // --mdp-backbone (Req 4.9, 5.1).
    const std::string bb_s = getOption(args, "--mdp-backbone", "resnet18");
    auto bb_opt = cfm::mdpBackboneTypeFromString(bb_s);
    if (!bb_opt) {
        std::cerr << "--mdp-backbone must be one of {lightweight_cnn, resnet18, mobilenet_v3_small}, got: " << bb_s << std::endl;
        return 2;
    }
    cfg.mdp_backbone = *bb_opt;

    // --mdp-backbone-weights.
    cfg.mdp_backbone_weights = getOption(args, "--mdp-backbone-weights", "models/pretrained/resnet18_in1k.pt");

    // --num-heads (Req 3.2).
    cfg.num_heads = std::stoi(getOption(args, "--num-heads", "4"));
    if (cfg.num_heads < 2) {
        std::cerr << "--num-heads must be >= 2, got: " << cfg.num_heads << std::endl;
        return 2;
    }

    // --seed (Req 6.1).
    cfg.seed = std::stoi(getOption(args, "--seed", "42"));

    auto result = cfm::CFMTrainer::train(cfg);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printTiming("cfm-train", elapsed, 0);
    if (!result) {
        std::cerr << "[cfm-train] failed: " << result.error_msg() << std::endl;
        return 1;
    }
    return 0;
#endif
}

int CLI::cmdCfmInfer(const std::vector<std::string>& args) {
    const auto start = std::chrono::steady_clock::now();
#ifndef CMS_HAS_LIBTORCH
    (void)args;
    std::cerr << "cfm-infer subcommand requires LibTorch. Rebuild with -DCMS_ENABLE_LIBTORCH=ON" << std::endl;
    return 1;
#else
    cfm::CFMInferenceConfig cfg;
    cfg.manifest_path     = getOption(args, "--manifest", "data/MS2/manifest_nir_rgb_val.csv");
    cfg.model_path        = getOption(args, "--model", "models/cfm.pt");
    cfg.output_dir        = getOption(args, "--output-dir", "output/cfm");
    cfg.num_disparities   = std::stoi(getOption(args, "--num-disparities", "96"));
    cfg.depth_min         = std::stod(getOption(args, "--depth-min", "1.0"));
    cfg.depth_max         = std::stod(getOption(args, "--depth-max", "80.0"));
    cfg.max_samples       = std::stoi(getOption(args, "--max-samples", "0"));
    const std::string use_gpu = getOption(args, "--use-gpu", "false");
    cfg.use_gpu           = (use_gpu == "true" || use_gpu == "1");

    // --attention (optional, validated against metadata).
    const std::string attn_s = getOption(args, "--attention", "");
    if (!attn_s.empty()) {
        auto attn_opt = cfm::attentionModeFromString(attn_s);
        if (!attn_opt) {
            std::cerr << "--attention must be one of {legacy_gated, real_scaled_dot}, got: " << attn_s << std::endl;
            return 2;
        }
        cfg.attention_mode = *attn_opt;
    }

    // --mdp-backbone (optional).
    const std::string bb_s = getOption(args, "--mdp-backbone", "");
    if (!bb_s.empty()) {
        auto bb_opt = cfm::mdpBackboneTypeFromString(bb_s);
        if (!bb_opt) {
            std::cerr << "--mdp-backbone must be one of {lightweight_cnn, resnet18, mobilenet_v3_small}, got: " << bb_s << std::endl;
            return 2;
        }
        cfg.mdp_backbone = *bb_opt;
    }

    cfg.mdp_backbone_weights = getOption(args, "--mdp-backbone-weights", "");

    // --num-heads (optional).
    const std::string nh_s = getOption(args, "--num-heads", "");
    if (!nh_s.empty()) {
        cfg.num_heads = std::stoi(nh_s);
    }

    auto result = cfm::CFMInference::run(cfg);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printTiming("cfm-infer", elapsed, 0);
    if (!result) {
        std::cerr << "[cfm-infer] failed: " << result.error_msg() << std::endl;
        return 1;
    }
    return 0;
#endif
}

void CLI::printUsage() const {
    std::cout << "Usage: cms <download|run|evaluate|visualize|analyze|train|cfm-train|cfm-infer|export-mdp-backbone|split-manifest> [options]\n";
}

int CLI::cmdSplitManifest(const std::vector<std::string>& args) {
    const auto start = std::chrono::steady_clock::now();

    const std::string input_s     = getOption(args, "--input");
    const std::string train_out_s = getOption(args, "--train-out");
    const std::string val_out_s   = getOption(args, "--val-out");
    const std::string train_size_s = getOption(args, "--train-size", "150");
    const std::string val_size_s   = getOption(args, "--val-size", "20");
    const std::string val_offset_s = getOption(args, "--val-offset", "8000");

    if (input_s.empty() || train_out_s.empty() || val_out_s.empty()) {
        std::cerr << "usage: cms split-manifest --input <path> --train-out <path> "
                     "--val-out <path> [--train-size <int|-1>] "
                     "[--val-size <int>] [--val-offset <int>]" << std::endl;
        return 2;
    }

    int train_size = 0;
    int val_size = 0;
    int val_offset = 0;
    try {
        train_size = std::stoi(train_size_s);
        val_size   = std::stoi(val_size_s);
        val_offset = std::stoi(val_offset_s);
    } catch (const std::exception& ex) {
        std::cerr << "split-manifest: invalid integer argument (" << ex.what() << ")" << std::endl;
        return 2;
    }

    auto result = splitManifest(std::filesystem::path(input_s),
                                std::filesystem::path(train_out_s),
                                std::filesystem::path(val_out_s),
                                train_size, val_size, val_offset);
    if (!result) {
        std::cerr << "split-manifest: " << result.error_msg() << std::endl;
        return 1;
    }

    // Report the row counts for quick verification (matches the Python
    // reference script's "Wrote N rows -> path" line).
    auto countRows = [](const std::filesystem::path& p) -> long {
        std::ifstream fin(p);
        if (!fin) return -1;
        long n = 0;
        std::string line;
        while (std::getline(fin, line)) ++n;
        return n - 1;  // subtract header
    };
    const long train_n = countRows(train_out_s);
    const long val_n   = countRows(val_out_s);
    std::cout << "Wrote " << train_n << " rows -> " << train_out_s << std::endl;
    std::cout << "Wrote " << val_n   << " rows -> " << val_out_s << std::endl;

    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printTiming("split-manifest", elapsed,
                static_cast<int>((train_n < 0 ? 0 : train_n) +
                                 (val_n   < 0 ? 0 : val_n)));
    return 0;
}

void CLI::printTiming(const std::string& command, double elapsed_sec, int sample_count) const {
    std::cout << command << " took " << elapsed_sec << " sec, samples=" << sample_count << std::endl;
}

int CLI::cmdExportMdpBackbone(const std::vector<std::string>& args) {
    const auto start = std::chrono::steady_clock::now();
#ifndef CMS_HAS_LIBTORCH
    (void)args;
    std::cerr << "export-mdp-backbone subcommand requires LibTorch. Rebuild with -DCMS_ENABLE_LIBTORCH=ON"
              << std::endl;
    return 1;
#else
    // `--help` short-circuits before any validation so users can discover
    // the flags without needing a real `.pth` file on disk. The text
    // mirrors design §3.2 and Req 4.8 — it spells out exactly which
    // submodule keys we extract for each backbone so users know what
    // ends up in the output archive.
    for (const auto& a : args) {
        if (a == "--help" || a == "-h") {
            std::cout
                << "Usage: cms export-mdp-backbone --kind {resnet18|mobilenet_v3_small}\n"
                << "                                --pth  <input .pth file>\n"
                << "                                --out  <output .pt archive>\n"
                << "\n"
                << "  Convert a TorchVision ImageNet checkpoint (a pickled state_dict\n"
                << "  saved by torch.save in Python) into a LibTorch archive whose\n"
                << "  keys are exactly the subset consumed by MDPBackbone.\n"
                << "\n"
                << "  Submodule keys retained:\n"
                << "    resnet18           : conv1.*, bn1.*, layer1.*\n"
                << "    mobilenet_v3_small : features.0.*, features.1.*\n"
                << "\n"
                << "  The first conv weight is collapsed from 3 to 1 input channel\n"
                << "  via .mean(dim=1, keepdim=true) so the resulting archive is\n"
                << "  ready to load into a single-channel MDPBackbone directly.\n"
                << "\n"
                << "  Download the TorchVision checkpoints manually with curl:\n"
                << "    curl -LO https://download.pytorch.org/models/resnet18-f37072fd.pth\n"
                << "    curl -LO https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth\n"
                << std::endl;
            return 0;
        }
    }

    const std::string kind    = getOption(args, "--kind");
    const std::string pth_s   = getOption(args, "--pth");
    const std::string out_s   = getOption(args, "--out");

    if (kind.empty() || pth_s.empty() || out_s.empty()) {
        std::cerr << "usage: cms export-mdp-backbone --kind {resnet18|mobilenet_v3_small} "
                     "--pth <input .pth> --out <output .pt>" << std::endl;
        return 2;
    }
    if (kind != "resnet18" && kind != "mobilenet_v3_small") {
        std::cerr << "export-mdp-backbone: invalid --kind '" << kind
                  << "' (expected 'resnet18' or 'mobilenet_v3_small')" << std::endl;
        return 2;
    }

    auto result = cfm::exportMdpBackbone(kind,
                                         std::filesystem::path(pth_s),
                                         std::filesystem::path(out_s));
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printTiming("export-mdp-backbone", elapsed, 0);
    if (!result) {
        std::cerr << "[export-mdp-backbone] failed: " << result.error_msg() << std::endl;
        return 1;
    }
    return 0;
#endif
}

}  // namespace cms
