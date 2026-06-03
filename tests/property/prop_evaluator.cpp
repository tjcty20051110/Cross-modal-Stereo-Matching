#include <cmath>
#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>
#include <vector>

#include "evaluation/evaluator.h"

RC_GTEST_PROP(EvaluatorProperty, InvalidPixelsDoNotAffectMetrics, (std::vector<float> pred_vals, std::vector<float> gt_vals)) {
    if (pred_vals.empty()) pred_vals.push_back(0.0f);
    if (gt_vals.empty()) gt_vals.push_back(0.0f);
    const size_t n = std::min<size_t>(std::min(pred_vals.size(), gt_vals.size()), 32);

    cv::Mat pred(1, static_cast<int>(n), CV_32F);
    cv::Mat gt(1, static_cast<int>(n), CV_32F);
    cv::Mat mask(1, static_cast<int>(n), CV_8U, cv::Scalar(0));
    for (size_t i = 0; i < n; ++i) {
        pred.at<float>(0, static_cast<int>(i)) = pred_vals[i];
        gt.at<float>(0, static_cast<int>(i)) = gt_vals[i];
        if (i % 2 == 0) mask.at<unsigned char>(0, static_cast<int>(i)) = 255;
    }

    auto baseline = cms::Evaluator::evaluate(pred, gt, mask, "base");
    cv::Mat modified = pred.clone();
    for (int i = 0; i < modified.cols; ++i) {
        if (mask.at<unsigned char>(0, i) == 0) modified.at<float>(0, i) += 1000.0f;
    }
    auto changed = cms::Evaluator::evaluate(modified, gt, mask, "changed");

    RC_ASSERT(baseline.is_valid);
    RC_ASSERT(changed.is_valid);
    RC_ASSERT(std::abs(baseline.mse - changed.mse) < 1e-6);
    RC_ASSERT(std::abs(baseline.epe - changed.epe) < 1e-6);
    RC_ASSERT(std::abs(baseline.d1_all - changed.d1_all) < 1e-6);
}
