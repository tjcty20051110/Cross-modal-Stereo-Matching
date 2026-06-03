# CFM 网络 — 基于论文 2411.03638 的跨模态立体深度估计

> 实现论文 *"Adaptive Stereo Depth Estimation with Multi-Spectral Images Across All Lighting Conditions"* (Qin et al., arXiv:2411.03638) 的四阶段跨模态深度估计架构。
>
> 本模块**与 ResNet18 Siamese 网络相互独立**，位于 `src/cfm/` 目录，对应 `cfm-train` 和 `cfm-infer` CLI 子命令。

## 一、架构概览

```
可见光图像 I_vis + NIR/热红外图像 I_thr
         ↓
┌─────────────────────────┐
│  CFM Module              │ ← 跨模态特征匹配 + 代价体构建
│   • FeatureExtractor×2    │
│   • CrossAttention×2      │
│   • Shifted Dot Product   │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  MDP Module              │ ← 各模态深度高斯分布 (μ, σ²)
│   • MDP_vis, MDP_thr      │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Degradation Masking    │ ← 用 vis-MDP 的不确定性过滤代价体
│   θ = μ + k·σ (k=1)      │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Depth Module            │ ← 融合 masked cost + 热红外特征
│   • 3×ConvBnReLU          │
│   • μ, log σ² 预测头       │
│   • 双线性上采样至全分辨率  │
└─────────────────────────┘
         ↓
     全分辨率深度图 D_map
```

## 二、各模块设计详解

### 2.1 CFM Module（Cross-modal Feature Matching）

**目标**：从 `I_vis` 和 `I_thr` 中提取像素对齐的特征，构建深度代价体。

**结构**：

1. **双分支 FeatureExtractor**（PSMNet 风格）
   - 输入：`[B, 1, H, W]`
   - 5 层 3×3 卷积 + 残差连接
   - 下采样至 H/4 × W/4（节省内存）
   - 输出：`[B, 32, H/4, W/4]`

2. **CrossAttention**（双向）
   - Vis→Thr：thr 特征作为 query，vis 特征作为 key
   - Thr→Vis：vis 特征作为 query，thr 特征作为 key
   - 用 channel-wise gated fusion 实现（原论文为完整 cross-attention，但显存受限时采用门控融合近似，保留跨模态信息流且显存开销恒定）

3. **代价体构建**（paper Eq. 3）
   - 对每个视差候选 `d_k ∈ {0, 1, ..., 47}`
   - `cost(u, v, d_k) = f_thr(u,v) · f_vis(u − d_k, v)`
   - 特征先 L2 归一化，点积 = 余弦相似度
   - 输出：`[B, 48, H/4, W/4]`

### 2.2 MDP Module（Modality-specific Depth Probability）

**目标**：对每个模态单独估计像素级高斯深度分布 `N(μ_uv, σ²_uv)`，用于 Degradation Masking 和最终深度融合。

**结构**：
- 轻量级编码器：3 层下采样卷积（输出 H/4 × W/4）+ 2 层 refine block
- **双输出头**：mean head 预测 `μ`，log_var head 预测 `log σ²`（用 log 保证数值稳定）
- **feat 输出**：最后一层 32 通道特征，用于 Depth Module 融合

论文使用 MaGNet 的 D-Net 作为 MDP 骨干；本实现采用轻量级近似以适应 8GB 显存。

### 2.3 Degradation Masking（paper Eq. 5-6）

**目标**：在可见光不可靠区域（低光、阴影、镜面反射）屏蔽代价体，让最终深度回落到单目热红外估计。

**算法**：
```
对每个像素 (u,v) 和视差候选 d_k:
  1. 查询 vis-MDP 的 Gaussian: P(d_k|u,v) = exp(−0.5·((d_k − μ)/σ)²)
  2. 阈值: θ = exp(−0.5·k²), k=1 (± 1σ 内的候选保留)
  3. soft gate: mask = sigmoid(20·(P − θ))  (保持梯度)
  4. masked_cost = cost × mask
```

**效果**：
- 可见光可靠区域（小 σ）→ mask ≈ 1 → 代价体原样保留
- 可见光不可靠区域（大 σ）→ mask ≈ 0 → 代价体被抑制，Depth Module 仅能依赖热红外特征

### 2.4 Depth Module（paper Eq. 7）

**目标**：将 masked cost volume 与 thermal MDP 特征融合，回归最终全分辨率深度 + 不确定性。

**结构**：
1. Concatenate: `[masked_cost (48 ch), thermal_feat (32 ch)] → 80 ch`
2. 3 层 ConvBnReLU： 80 → 64 → 64 → 32
3. 双输出头：`μ_d` 和 `log σ²_d`
4. 双线性上采样：`H/4 × W/4 → H × W`

## 三、三阶段训练（paper Section III.E）

论文将训练分为三个阶段，降低端到端优化难度：

| 阶段 | 可训练模块 | 损失函数 | 说明 |
|------|-----------|----------|------|
| **CFM** | CFM Module + Depth Module | L1 on expected depth (Eq. 8) | 让代价体对应正确视差 |
| **MDP** | MDP_vis + MDP_thr | Negative Log-Likelihood (Eq. 9) | 各模态独立高斯深度 |
| **Depth** | Depth Module (冻结 CFM/MDP) | NLL on final depth (Eq. 9) | 融合阶段，最终回归 |
| **ALL** | 所有模块 | 三者加权组合 | 简化一站式训练 |

本实现提供 `--stage {cfm,mdp,depth,all}` 开关。

## 四、复现流程

### 4.1 前置条件

与 ResNet18 Siamese 相同，需要构建支持 LibTorch 的可执行文件：

```bash
# 构建（以 CUDA 12.1 + LibTorch cu121 为例）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMS_BUILD_TESTS=OFF \
      -DCMS_ENABLE_LIBTORCH=ON \
      -DTorch_DIR=$HOME/libtorch_cu121/libtorch/share/cmake/Torch \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc
cmake --build build --target cms -j$(nproc)

# 生成跨模态清单（参考 NIR_left + RGB_right）
python3 scripts/generate_manifest.py --data-root data/MS2 \
        --output data/MS2/manifest_nir_rgb.csv --mode nir_rgb
python3 scripts/split_manifest.py --input data/MS2/manifest_nir_rgb.csv \
        --train-out data/MS2/manifest_nir_rgb_train.csv \
        --val-out   data/MS2/manifest_nir_rgb_val.csv \
        --train-size 200 --val-size 10 --val-offset 8000
```

### 4.2 训练

**方案 A：一站式训练（推荐快速验证）**

```bash
./build/bin/cms cfm-train \
    --manifest data/MS2/manifest_nir_rgb_train.csv \
    --output   models/cfm_all.pt \
    --stage all \
    --epochs 10 \
    --batch-size 4 \
    --lr 0.0001 \
    --num-disparities 48 \
    --depth-min 1.0 --depth-max 80.0 \
    --log-every 20 \
    --use-gpu true
```

三种损失加权组合：
```
loss = L_final + 0.5 · L_cfm + 0.25 · (L_mdp_vis + L_mdp_thr)
```

**方案 B：三阶段串行训练（更稳定，接近论文设置）**

```bash
# Stage 1: CFM（学会代价体匹配）
./build/bin/cms cfm-train --manifest data/MS2/manifest_nir_rgb_train.csv \
    --output models/cfm_s1.pt --stage cfm --epochs 5 --batch-size 4 --use-gpu true

# Stage 2: MDP（在 stage 1 之上继续训 MDP）
./build/bin/cms cfm-train --manifest data/MS2/manifest_nir_rgb_train.csv \
    --output models/cfm_s2.pt --stage mdp --epochs 5 --batch-size 4 --use-gpu true \
    --init-from models/cfm_s1.pt

# Stage 3: Depth（冻结前两阶段，训融合模块）
./build/bin/cms cfm-train --manifest data/MS2/manifest_nir_rgb_train.csv \
    --output models/cfm.pt --stage depth --epochs 5 --batch-size 4 --use-gpu true \
    --init-from models/cfm_s2.pt
```

### 4.3 推理

```bash
./build/bin/cms cfm-infer \
    --manifest data/MS2/manifest_nir_rgb_val.csv \
    --model    models/cfm.pt \
    --output-dir output/cfm_pred \
    --num-disparities 48 \
    --depth-min 1.0 --depth-max 80.0 \
    --use-gpu true
```

输出文件：
- `{sample_id}_depth.png` — uint16 深度图，`depth_meter = pixel / 256.0`
- `{sample_id}_disp.png` — uint16 视差图（从深度计算），用于 `cms evaluate` 兼容
- `{sample_id}_mask.png` — 有效像素掩码

### 4.4 评估

因为 CFM 推理输出的 `_disp.png` 与传统流水线格式兼容，可直接复用 `cms evaluate`：

```bash
./build/bin/cms evaluate \
    --config   configs/nir_rgb.yaml \
    --manifest data/MS2/manifest_nir_rgb_val.csv \
    --pred-dir output/cfm_pred \
    --csv      output/cfm_pred/metrics.csv
```

### 4.5 可视化

```bash
./build/bin/cms visualize \
    --config   configs/nir_rgb.yaml \
    --manifest data/MS2/manifest_nir_rgb_val.csv \
    --pred-dir output/cfm_pred \
    --output-dir output/cfm_pred/viz
```

## 五、命令行参数总表

### `cfm-train`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--manifest` | `data/MS2/manifest_nir_rgb_train.csv` | 训练清单 CSV |
| `--output` | `models/cfm.pt` | 模型输出路径 |
| `--stage` | `all` | `cfm` / `mdp` / `depth` / `all` |
| `--epochs` | `5` | 训练轮数 |
| `--batch-size` | `4` | 梯度累积步数（单图前向） |
| `--lr` | `0.0001` | Adam 学习率 |
| `--num-disparities` | `48` | 代价体视差候选数 |
| `--depth-min` | `1.0` | 深度搜索下限（米） |
| `--depth-max` | `80.0` | 深度搜索上限（米） |
| `--max-samples` | `0` | 训练集上限（0=全部） |
| `--log-every` | `10` | 日志间隔 |
| `--use-gpu` | `true` | 启用 CUDA |
| `--init-from` | *(空)* | 从检查点初始化 |

### `cfm-infer`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--manifest` | `data/MS2/manifest_nir_rgb_val.csv` | 推理清单 |
| `--model` | `models/cfm.pt` | 模型路径 |
| `--output-dir` | `output/cfm` | 输出目录 |
| `--num-disparities` | `48` | 必须与训练一致 |
| `--depth-min` | `1.0` | 必须与训练一致 |
| `--depth-max` | `80.0` | 必须与训练一致 |
| `--max-samples` | `0` | 推理样本数 |
| `--use-gpu` | `false` | 启用 CUDA |

## 六、实现细节与性能

### 6.1 参数量

| 模块 | 参数量 (约) |
|------|------------|
| FeatureExtractor × 2 | 90k × 2 = 180k |
| CrossAttention × 2 | 6k × 2 = 12k |
| MDP Module × 2 | 50k × 2 = 100k |
| Depth Module | 70k |
| **总计** | **~360k** (≈1.5 MB 模型文件) |

比 ResNet18 Siamese (1.3M) 更轻量，但架构复杂度更高。

### 6.2 显存占用（RTX 4060, 8GB）

| 阶段 | 前向 + 反向显存 |
|------|----------------|
| batch_size = 1 | ~1.8 GB |
| batch_size = 4 (梯度累积) | ~1.8 GB (同前) |
| batch_size = 4 (真实) | ~7.2 GB (接近上限) |

**推荐**：`--batch-size 4` 用梯度累积，避免 OOM。

### 6.3 训练速度（RTX 4060）

| 阶段 | 每图耗时 | 200 样本 × 5 epoch |
|------|----------|---------------------|
| CFM only | ~0.4 s | ~7 分钟 |
| MDP only | ~0.2 s | ~4 分钟 |
| Depth only | ~0.3 s | ~5 分钟 |
| ALL（联合） | ~0.5 s | ~9 分钟 |

## 七、与 ResNet18 Siamese 的区别

| 维度 | ResNet18 Siamese | CFM Pipeline |
|------|------------------|--------------|
| **输出** | 特征图（配合 SGM 做视差） | 端到端深度图 |
| **损失** | Triplet Margin Loss | L1 + NLL (高斯负对数似然) |
| **架构** | 单一 CNN | 4 阶段组合 (CFM + 2×MDP + Depth) |
| **退化处理** | 无 | Degradation Masking |
| **训练数据需求** | 只需 GT 视差 | 需要 GT 深度（通过焦距+基线从视差转换） |
| **推理流水线** | 特征 → 代价体 → SGM → 视差 | 端到端直接输出深度 |
| **适用场景** | 标准跨模态立体匹配 | 光照变化剧烈场景（夜间、雨天等） |
| **代码位置** | `src/training/` | `src/cfm/` |
| **CLI 子命令** | `cms train` / `cms run --config dl_feature.yaml` | `cms cfm-train` / `cms cfm-infer` |

两套网络**完全独立**，可以并行开发、独立评估。

## 八、源码结构

```
src/cfm/
├── cfm_network.h        # CFMModule / MDPModule / DepthModule / CFMPipeline
├── cfm_network.cpp      # 四个模块的前向实现 + Degradation Masking
├── cfm_trainer.h        # TrainingConfig + TrainStage 枚举
├── cfm_trainer.cpp      # 三阶段训练循环 + 损失函数 (L1 / NLL)
├── cfm_inference.h      # InferenceConfig
└── cfm_inference.cpp    # 推理循环 + depth→disparity 后处理
```

CLI 入口位于 `src/cli/cli.cpp` 的 `cmdCfmTrain` / `cmdCfmInfer` 函数。

## 九、已知限制与改进方向

1. **CrossAttention 近似**：当前用 channel-wise gated fusion 替代完整 attention，在 8GB 显存下更安全。若有更大显存（≥16GB），可实现完整的 Q·K^T softmax。
2. **MDP 骨干**：论文使用 MaGNet D-Net（ImageNet 预训练），本实现为轻量版。若引入预训练权重，精度应进一步提升。
3. **深度候选采样**：当前为逆深度均匀采样（1~80m）。可根据数据集调整。
4. **论文指标**：论文报告 Abs Rel / RMSE / δ<1.25 等单目深度指标。本实现通过 depth → disparity 转换后用 MSE / EPE / D1-all 评估，便于与 ResNet18 Siamese 和 Census 对比。若需复现论文指标，需额外实现深度评估器。

## 十、参考

- 论文 PDF：`2411.03638v1/main.md`
- ResNet18 Siamese 设计：`docs/neural_network_README.md`
- 跨模态实验报告：`docs/cross_modal_experiment/cross_modal_experiment_report.md`
