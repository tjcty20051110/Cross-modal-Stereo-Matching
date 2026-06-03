# 跨模态立体匹配系统 — 运行指南

## 环境依赖

| 依赖 | 最低版本 | 用途 |
|------|----------|------|
| CMake | 3.18 | 构建系统 |
| GCC / Clang | C++17 支持 | 编译器 |
| OpenCV | 4.0 | 图像处理、I/O |
| yaml-cpp | — | 配置文件解析 |
| Python | 3.8+ | 清单文件生成脚本 |
| NumPy | — | 读取 calib.npy |

## 构建步骤

```bash
cd cross-modal-stereo-matching

# 配置（Release 模式，不编译测试以加快构建）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMS_BUILD_TESTS=OFF

# 编译
cmake --build build --target cms -j$(nproc)
```

编译产物位于 `build/bin/cms`。

## 数据准备

### 1. 数据集目录结构

程序期望 MS2 数据集按以下结构组织（用户已手动处理）：

```
data/MS2/
├── sync_data/
│   ├── rgb/
│   │   ├── img_left/       # RGB 左图 (PNG, 384×1224)
│   │   └── img_right/      # RGB 右图 (PNG, 384×1224)
│   ├── nir/
│   │   ├── img_left/       # NIR 左图 (PNG, 352×1280)
│   │   └── img_right/      # NIR 右图 (PNG, 352×1280)
│   └── calib.npy           # 标定参数 (NumPy 格式)
├── proj_depth/
│   └── rgb/
│       └── depth_filtered/ # 深度真值 (uint16 PNG, depth = pixel/256.0 米)
├── manifest.csv            # 同模态清单 (自动生成)
├── manifest_cross.csv      # 跨模态清单 (自动生成)
├── calib_rgb.yaml          # 同模态标定参数 (自动生成)
└── calib_cross.yaml        # 跨模态标定参数 (自动生成)
```

### 2. 生成清单文件

运行 Python 脚本从数据集目录自动生成清单 CSV 和标定 YAML：

```bash
# 同模态模式 (RGB left + RGB right, 基线 ~299mm)
python3 scripts/generate_manifest.py \
    --data-root data/MS2 \
    --output data/MS2/manifest.csv \
    --mode rgb

# 跨模态模式 (RGB left + NIR left, 基线 ~54mm)
python3 scripts/generate_manifest.py \
    --data-root data/MS2 \
    --output data/MS2/manifest_cross.csv \
    --mode cross
```

可选参数：
- `--max-samples N`：限制样本数量（用于快速测试，0 表示全部）
- `--train-ratio 0.8`：训练集比例

脚本会自动：
- 扫描图像目录，按文件名匹配左图、右图、深度真值
- 从 `calib.npy` 提取焦距和基线，写入 `calib.yaml`
- 生成包含相对路径的 CSV 清单

## 运行立体匹配

### 同模态立体匹配（推荐，精度较高）

```bash
./build/bin/cms run \
    --config configs/default.yaml \
    --manifest data/MS2/manifest.csv \
    --output-dir output/rgb_stereo
```

### 跨模态立体匹配

```bash
./build/bin/cms run \
    --config configs/cross_modal.yaml \
    --manifest data/MS2/manifest_cross.csv \
    --output-dir output/cross_modal
```

### 快速测试（少量样本）

```bash
# 先生成少量样本的清单
python3 scripts/generate_manifest.py \
    --data-root data/MS2 \
    --output data/MS2/manifest_quick.csv \
    --mode rgb \
    --max-samples 10

# 运行
./build/bin/cms run \
    --config configs/default.yaml \
    --manifest data/MS2/manifest_quick.csv \
    --output-dir output/quick_test
```

### 输出文件

运行完成后，输出目录包含：
- `{sample_id}_disp.png` — 视差图（uint16，实际视差 = pixel / 256.0）
- `{sample_id}_mask.png` — 可信度掩码（0=无效，255=有效）

## 评估指标

对已有的预测结果计算 MSE、EPE、D1-all 指标：

```bash
./build/bin/cms evaluate \
    --config configs/default.yaml \
    --manifest data/MS2/manifest.csv \
    --pred-dir output/rgb_stereo \
    --csv output/rgb_stereo/metrics.csv
```

输出示例：
```
MSE=239.308, EPE=10.5564, D1-all=70.1665
```

指标说明：
- **MSE**：均方误差（像素²）
- **EPE**：端点误差，即平均绝对视差误差（像素）
- **D1-all**：错误匹配率，|pred-gt| > max(3px, 5%×gt) 的像素比例（%）

## 可视化

生成伪彩色视差图、误差热力图和对比大图：

```bash
./build/bin/cms visualize \
    --config configs/default.yaml \
    --manifest data/MS2/manifest.csv \
    --pred-dir output/rgb_stereo \
    --output-dir output/rgb_stereo/visualizations
```

输出文件：
- `{sample_id}_pred.png` — 预测视差伪彩色图
- `{sample_id}_gt.png` — 真值视差伪彩色图
- `{sample_id}_error.png` — 误差热力图
- `{sample_id}_comparison.png` — 四宫格对比大图

## 模态差异分析

对跨模态数据进行统计分析，比较不同代价度量的效果：

```bash
./build/bin/cms analyze \
    --config configs/default.yaml \
    --manifest data/MS2/manifest.csv \
    --output-dir output/analysis
```

输出：
- `modality_stats.csv` — 逐样本亮度/梯度统计
- `cost_metric_comparison.csv` — Census vs NCC 代价度量对比
- `modality_summary.png` — 汇总可视化图
- `modality_analysis_report.md` — 分析报告

## 深度学习特征（可选，需启用 LibTorch）

项目支持通过训练一个 Siamese CNN 提取模态无关特征，替代 Census 变换以提升跨模态匹配精度。详细的网络设计与训练原理见 [`docs/neural_network_README.md`](neural_network_README.md)。

### 启用 LibTorch 构建

```bash
# 一次性下载 CPU 版 LibTorch（约 200MB）
wget -O /tmp/libtorch.zip \
    'https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip'
unzip -q /tmp/libtorch.zip -d ~/libtorch_cpu/

# 重新配置并编译
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMS_BUILD_TESTS=OFF \
      -DCMS_ENABLE_LIBTORCH=ON \
      -DTorch_DIR=$HOME/libtorch_cpu/libtorch/share/cmake/Torch
cmake --build build --target cms -j$(nproc)
```

### 训练

```bash
./build/bin/cms train \
    --manifest data/MS2/manifest.csv \
    --output models/siamese.pt \
    --epochs 5 \
    --batch-size 2 \
    --samples-per-image 512 \
    --max-samples 500
```

### 使用训练好的模型推理

```bash
./build/bin/cms run \
    --config configs/dl_feature.yaml \
    --manifest data/MS2/manifest.csv \
    --output-dir output/dl
```

## 配置参数说明

### configs/default.yaml（同模态）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_disparity` | 128 | 最大视差搜索范围（像素） |
| `feature_method` | CENSUS | 特征提取方法 |
| `cost_metric` | CENSUS_HAMMING | 匹配代价度量 |
| `aggregation_method` | SGM | 代价聚合方法 |
| `sgm.num_directions` | 8 | SGM 扫描方向数 |
| `sgm.p1` | 10 | 小视差变化惩罚 |
| `sgm.p2` | 120 | 大视差变化惩罚 |

### configs/cross_modal.yaml（跨模态）

跨模态配置使用较小的 `max_disparity=32`（因为基线仅 54mm），以及较低的 SGM 惩罚参数。

## 性能参考

在当前硬件上的典型运行时间（384×1224 图像，SGM 8方向）：

| 配置 | 每帧耗时 | 说明 |
|------|----------|------|
| 同模态 (D=128) | ~13 秒 | RGB left + RGB right |
| 跨模态 (D=32) | ~3 秒 | RGB left + NIR left |

## 完整工作流示例

```bash
# 1. 构建
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMS_BUILD_TESTS=OFF
cmake --build build --target cms -j$(nproc)

# 2. 生成清单（首次运行或数据变更后）
python3 scripts/generate_manifest.py --data-root data/MS2 --output data/MS2/manifest.csv --mode rgb

# 3. 运行立体匹配（可用 --max-samples 限制数量进行快速测试）
./build/bin/cms run --config configs/default.yaml --output-dir output/rgb_stereo

# 4. 评估
./build/bin/cms evaluate --config configs/default.yaml --pred-dir output/rgb_stereo

# 5. 可视化
./build/bin/cms visualize --config configs/default.yaml --pred-dir output/rgb_stereo --output-dir output/rgb_stereo/viz
```

## 常见问题

**Q: 运行时报 "Manifest file does not exist"**
A: 需要先运行 `python3 scripts/generate_manifest.py` 生成清单文件。

**Q: 跨模态匹配精度很低**
A: 跨模态基线仅 54mm，有效视差范围很小（最大约 7-8 像素），这是数据集本身的限制。同模态 RGB 立体对（基线 299mm）能获得更好的精度。

**Q: 如何只处理部分样本？**
A: 使用 `--max-samples N` 参数生成清单时限制样本数量，或直接编辑 CSV 文件。

**Q: 内存不足**
A: 代价体占用约 `H × W × D × 4` 字节。对于 384×1224×128，约 230MB。可降低 `max_disparity` 或处理更小的图像。
