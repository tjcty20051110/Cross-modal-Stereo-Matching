// trainer.h — Training driver for the Siamese cross-modal feature network
#pragma once

#ifdef CMS_HAS_LIBTORCH

#include <filesystem>
#include <string>

#include "common/types.h"
#include "data/data_loader.h"

namespace cms {

/// Training hyperparameters for the Siamese network.
struct TrainingConfig {
    std::filesystem::path manifest_path;        ///< Manifest CSV for training data
    std::filesystem::path output_model_path;    ///< Path to save trained TorchScript model
    int num_epochs = 5;                          ///< Number of training epochs
    int batch_size = 2;                          ///< Images processed per batch
    int samples_per_image = 512;                 ///< Number of pixel triplets per image
    int patch_half = 2;                          ///< Half window for negative sample exclusion
    int neg_min_offset = 4;                      ///< Minimum negative-sample offset (pixels)
    int neg_max_offset = 64;                     ///< Maximum negative-sample offset (pixels)
    double learning_rate = 1e-3;                 ///< Adam learning rate
    double margin = 0.2;                          ///< Triplet-loss margin
    int max_samples = 0;                          ///< Cap on manifest samples (0 = all)
    int log_every = 10;                           ///< Log cadence (in batches)
    bool use_gpu = true;                          ///< Use CUDA if available
};

/// Trainer that fits a Siamese network on cross-modal stereo triplets.
class Trainer {
public:
    /// Train the Siamese feature network. Saves a TorchScript model on success.
    /// @param config Training configuration
    /// @return Success or error message
    static Result<void> train(const TrainingConfig& config);
};

} // namespace cms

#endif // CMS_HAS_LIBTORCH
