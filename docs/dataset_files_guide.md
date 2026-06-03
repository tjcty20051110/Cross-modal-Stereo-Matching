# MS2 数据集文件使用指南

本文档说明在跨模态立体匹配项目中，MS2 数据集的哪些文件是项目所需要的。

## 数据集概览

当前已下载的数据集位于 `data/MS2/`，包含序列 `_2021-08-06-10-59-33`，分为三个子集：

| 子集 | 用途 | 是否需要 |
|------|------|----------|
| `sync_data/` | 同步的多模态立体图像对 + 标定参数 | **需要** |
| `proj_depth/` | LiDAR 投影深度图（深度真值） | **需要** |
| `odem/` | 里程计位姿（用于多帧融合等） | 不需要 |

## 项目所需文件

### 1. 立体图像对（核心输入）

位于 `sync_data/{sequence}/` 下：

| 路径 | 内容 | 格式 | 项目角色 |
|------|------|------|----------|
| `rgb/img_left/XXXXXX.png` | RGB 左图 | PNG | 跨模态立体对的一侧（左图） |
| `rgb/img_right/XXXXXX.png` | RGB 右图 | PNG | 同模态参考 / 可视化 |
| `nir/img_left/XXXXXX.png` | NIR 左图 | PNG | 跨模态立体对的一侧 |
| `nir/img_right/XXXXXX.png` | NIR 右图 | PNG | 跨模态立体对的另一侧（右图） |

本项目的跨模态立体匹配任务：以 **RGB 左图** 作为参考图像，以 **NIR 右图**（或同模态 RGB 右图）作为匹配图像，构成跨模态立体对。

### 2. 标定参数

| 路径 | 内容 | 格式 | 项目角色 |
|------|------|------|----------|
| `sync_data/{sequence}/calib.npy` | 所有相机的内参、旋转、平移矩阵 | NumPy .npy | 提供焦距、基线距离，用于视差→深度转换 |

`calib.npy` 中与本项目相关的关键参数：
- `K_rgbL`, `K_rgbR` — RGB 立体对的校正内参矩阵
- `T_rgbR` — RGB 右相机相对左相机的平移（基线，mm 单位）
- `K_nirL`, `K_nirR` — NIR 立体对的校正内参矩阵
- `T_nirR` — NIR 右相机相对左相机的平移
- `R_nir2rgb`, `T_nir2rgb` — NIR 左到 RGB 左的外参（跨模态对齐）

### 3. 深度真值（Ground Truth）

位于 `proj_depth/{sequence}/` 下：

| 路径 | 内容 | 格式 | 项目角色 |
|------|------|------|----------|
| `rgb/depth_filtered/XXXXXX.png` | RGB 视角的滤波深度图 | uint16 PNG | **推荐真值**，多帧融合后滤波，质量较高 |
| `rgb/depth/XXXXXX.png` | RGB 视角的单帧深度图 | uint16 PNG | 备选真值，稀疏但无噪声 |
| `nir/depth_filtered/XXXXXX.png` | NIR 视角的滤波深度图 | uint16 PNG | NIR 视角评估时使用 |
| `nir/depth/XXXXXX.png` | NIR 视角的单帧深度图 | uint16 PNG | 备选 |

深度图读取方式：`depth_meter = imread(path, UNCHANGED) / 256.0`（uint16 存储，除以 256 得到米制深度）。

## 不需要的文件

| 路径 | 原因 |
|------|------|
| `sync_data/{sequence}/thr/` | 热红外图像，本项目仅使用 RGB + NIR |
| `sync_data/{sequence}/lidar/` | 原始 LiDAR 点云，已通过 proj_depth 投影为深度图 |
| `sync_data/{sequence}/gps_imu/` | GPS/IMU 数据，立体匹配不需要 |
| `odem/` | 里程计位姿数据，本项目不涉及多帧融合或 SLAM |
| `proj_depth/{sequence}/rgb/depth_multi/` | 多帧合并深度（噪声大），不推荐 |
| `proj_depth/{sequence}/rgb/intensity/` | LiDAR 反射强度图，非深度真值 |
| `proj_depth/{sequence}/rgb/intensity_multi/` | 同上 |
| `proj_depth/{sequence}/thr/` | 热红外视角深度，不使用 |
| `*.tar.bz2` | 压缩包原始文件，已解压，可删除节省空间 |

## 典型跨模态立体对构成

对于帧号 `N`（如 `000100`）：

```
左图（参考）: sync_data/rgb/img_left/000100.png
右图（匹配）: sync_data/nir/img_left/000100.png   (跨模态, 基线~54mm)
             或 sync_data/rgb/img_right/000100.png  (同模态, 基线~299mm)
深度真值:     proj_depth/rgb/depth_filtered/000100.png
标定文件:     sync_data/calib.npy
```

## 数据规模

当前序列包含约 10,442 帧同步数据。

## 注意事项

1. 所有图像已完成立体校正（rectified），可直接用于水平方向的视差搜索
2. 深度图中值为 0 的像素表示无效（无 LiDAR 投影覆盖）
3. 标定文件中的平移矩阵单位为 **毫米**，使用时需转换为米
4. RGB 与 NIR 相机之间存在外参偏移，跨模态匹配时需注意基线计算
