#include <cstdlib>
#include <gtest/gtest.h>
#include <rapidcheck/gtest.h>

#include "features/feature_extractor.h"

RC_GTEST_PROP(FeatureExtractorProperty, CensusShapeConsistency, (int rows_raw, int cols_raw)) {
    const int rows = std::abs(rows_raw) % 16 + 1;
    const int cols = std::abs(cols_raw) % 16 + 1;
    cms::CensusFeatureExtractor extractor(9);
    cv::Mat image(rows, cols, CV_32F, cv::Scalar(0.5f));
    auto result = extractor.extract(image);
    RC_ASSERT(result.has_value());
    RC_ASSERT(result.value().data.rows == rows);
    RC_ASSERT(result.value().data.cols == cols);
    RC_ASSERT(result.value().channels == 80);
    RC_ASSERT(result.value().downsample_factor == 1);
}
