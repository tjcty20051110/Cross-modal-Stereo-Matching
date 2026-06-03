// cfm_trainer.h — Training driver for the CFM pipeline (paper 2411.03638)
#pragma once

#ifdef CMS_HAS_LIBTORCH

#include <filesystem>
#include "cfm/cfm_attention.h"
#include "cfm/mdp_backbone.h"
#include "common/types.h"

namespace cms {
namespace cfm {

enum class TrainStage {
    CFM,    ///< Stage 1: train CFM Module (cost-volume + L1 depth)
    MDP,    ///< Stage 2: train vis/thr MDP Modules (NLL depth loss)
    DEPTH,  ///< Stage 3: train Depth Module with frozen CFM/MDP
    ALL,    ///< Convenience: run stages 1 → 2 → 3 sequentially
};

struct CFMTrainingConfig {
    std::filesystem::path manifest_path;
    std::filesystem::path output_model_path;

    TrainStage stage = TrainStage::ALL;
    int    num_epochs = 5;
    int    batch_size = 1;      ///< Gradient accumulation steps
    double learning_rate = 1e-4;
    int    num_disparities = 96;
    double depth_min = 1.0;
    double depth_max = 80.0;
    int    max_samples = 0;     ///< 0 = all
    int    log_every = 10;
    bool   use_gpu = true;
    /// If non-empty and file exists, load this checkpoint before training
    /// (used to initialize stages 2/3 from stage-1 weights).
    std::filesystem::path init_from;

    // New fields (Req 3, 4, 5, 6)
    AttentionMode   attention_mode = AttentionMode::RealScaledDot;
    MDPBackboneType mdp_backbone   = MDPBackboneType::ResNet18;
    std::filesystem::path mdp_backbone_weights = "models/pretrained/resnet18_in1k.pt";
    int num_heads = 4;
    int seed      = 42;
};

class CFMTrainer {
public:
    static Result<void> train(const CFMTrainingConfig& config);
};

} // namespace cfm
} // namespace cms

#endif // CMS_HAS_LIBTORCH
