# 可视化改进总结

本文档总结了跨模态立体匹配项目中最终可视化效果的改进内容，供项目报告与代码审阅使用。

## 目标

原始最终可视化效果难以阅读：

- 对比图采用简单的 2×2 网格，缺乏视觉层次感。
- 文字标签直接绘制在图像内容上，在明亮区域变得无法辨认。
- 真值视差是稀疏的 LiDAR 投影，在图上表现为大量黑色背景中的微小彩点。
- 误差热力图过于稀疏，视觉上充满噪声。
- 视差归一化可能被离群值主导，导致有用区域坍缩为相似颜色。

本次改进的重点在于：使最终输出图像成为可读的实验结果图。

## 修改文件

- `src/visualization/visualizer.cpp`
- `src/visualization/visualizer.h`
- `src/cli/cli.cpp`
- `tests/unit/test_visualizer.cpp`

## 主要改动

### 1. 鲁棒的视差颜色缩放

视差渲染器不再仅依赖原始最大值，而是计算一个鲁棒的正值范围。

实现思路：

- 收集所有有限且大于零的有效视差像素。
- 使用有效值的高分位数作为实际显示范围。
- 将该范围与 `config.max_disparity` 进行钳制。
- 将视差图转换为 `CV_8U`，并应用 OpenCV 的 `COLORMAP_TURBO`。

收益：

- 即使存在少数离群像素，局部视差细节仍然可见。
- 预测视差图更高效地利用了可用的颜色范围。

### 2. 稀疏真值增强

MS2 真值是稀疏的，因为它来自投影深度/LiDAR 数据。直接渲染会导致其几乎不可见。

实现思路：

- 根据正值视差构建有效掩码。
- 估计有效像素比例。
- 若图像稀疏，则使用小型椭圆形核对有效掩码和着色后的视差图同时进行膨胀操作。
- 保持无效区域为暗色，同时使有效点/扫描线足够粗以便肉眼观察。

收益：

- 稀疏真值在不影响底层数值数据的前提下变得可读。
- 道路、边界和深度层次的视觉结构更加清晰。

### 3. 改进的误差热力图

误差图现在仅在有效真值像素上显示绝对视差误差。

实现思路：

- 仅在 `valid_mask` 非零的位置计算 `abs(pred - gt)`。
- 对误差值使用鲁棒的分位数缩放。
- 应用更适合误差幅度可视化的 OpenCV `COLORMAP_INFERNO`。
- 对于稀疏掩码，以与稀疏真值相同的方式膨胀可见的误差点。

收益：

- 误差图清晰展示了预测与真值偏离的位置。
- 无效区域不再主导整张图。

### 4. 仪表盘式最终对比图

最终的 `*_comparison.png` 从简单的 2×2 网格重新设计为结构化的 2×3 仪表盘。

新布局：

1. `Left RGB`（左 RGB）
2. `Right IR`（右红外）
3. `Prediction overlay`（预测叠加图）
4. `Predicted disparity`（预测视差）
5. `Ground truth`（真值）
6. `Absolute error`（绝对误差）

每个面板包含：

- 深色标题栏。
- 具有稳定宽高比的内容区域。
- 底部副标题文字。
- 可选的带像素单位的颜色条。
- 一致的边距、背景和边框。

收益：

- 最终可视化效果适合用于报告和演示。
- 预测、真值和误差可一目了然地进行对比。

### 5. 预测叠加图

新增了一个叠加面板，将预测视差颜色图混合到左侧 RGB 图像上。

实现思路：

- 将两张图像均转换为 BGR 显示格式。
- 将预测颜色图缩放到左图尺寸。
- 在 RGB 图像上混合有效视差颜色。
- 保留无效黑色区域不变。

收益：

- 更容易判断视差预测是否与道路边缘、树木和车辆等场景结构对齐。

### 6. CLI 集成

`visualize` 子命令现在将生成的误差热力图传递给 `Visualizer::renderComparison`。

旧行为：

- 对比图仅包含左图、右图、预测图和真值。

新行为：

- 对比图将预测叠加图和绝对误差作为最终仪表盘的一部分纳入其中。

### 7. 新增测试

在 `tests/unit/test_visualizer.cpp` 中新增了两项测试：

- `SparseDisparityPointsAreExpandedForReadability`
- `ComparisonWithErrorBuildsDashboardCanvas`

这些测试验证：

- 稀疏的有效视差点在视觉上被放大。
- 当提供误差热力图时，对比图渲染器会生成更大的仪表盘画布。

## 验证

相关改动已在 WSL Ubuntu 22.04 中验证。

使用的命令：

```bash
cmake -S . -B build-wsl
cmake --build build-wsl --target test_visualizer cms -j4
./build-wsl/bin/test_visualizer
```

结果：

```text
4 tests passed
```

此外，还使用真实样本进行了渲染：

```bash
./build-wsl/bin/cms visualize \
  --config configs/default.yaml \
  --manifest output/manifest_val_wsl.csv \
  --pred-dir output \
  --output-dir output/visualizations_wsl \
  008000_0
```

生成的对比图确认最终输出已符合预期的增强仪表盘风格。

## 在 Linux 上复现

在项目根目录下执行：

```bash
cmake -S . -B build
cmake --build build --target cms -j4

./build/bin/cms visualize \
  --config configs/default.yaml \
  --pred-dir output \
  --output-dir output/visualizations
```

最终增强后的图像生成路径为：

```text
output/visualizations/*_comparison.png
```

## 设计原则

代码不修改预测或评估数据，仅改进展示效果：

- 更鲁棒的显示缩放。
- 更好的稀疏数据可读性。
- 更清晰的视觉对比布局。
- 更适合报告的最终图像。
