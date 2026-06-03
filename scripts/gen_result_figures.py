#!/usr/bin/env python3
"""
生成跨模态立体匹配算法的结果对比图
参考 prj5_design 要求：算法结果可视化
"""

import matplotlib
matplotlib.use('Agg')  # 非GUI后端，避免Qt冲突
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

def load_disparity_color(path, max_disp=96):
    """加载视差图并转换为伪彩色"""
    disp = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if disp is None:
        return None

    if len(disp.shape) == 3:
        disp = disp[:, :, 0]

    disp = disp.astype(np.float32)

    # 归一化到0-max_disp范围
    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 应用伪彩色
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
    disp_color = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)

    return disp_color, disp

def load_ground_truth(gt_path, max_disp=96):
    """加载GT视差（深度图转视差）"""
    depth = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None

    if len(depth.shape) == 3:
        depth = depth[:, :, 0]

    # MS2数据集: depth_m = pixel / 256
    depth_float = depth.astype(np.float32) / 256.0

    # 深度转视差: disp = focal * baseline / depth
    focal = 638.9  # px
    baseline = 0.3529  # meters
    valid_mask = depth_float > 0.1

    disp = np.zeros_like(depth_float)
    disp[valid_mask] = (focal * baseline) / depth_float[valid_mask]
    disp = np.clip(disp, 0, max_disp)

    # 伪彩色
    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
    disp_color = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)

    return disp_color, disp

def add_label_border(img, label_text, color=(0, 255, 0)):
    """给图像添加标签边框"""
    h, w = img.shape[:2]
    border_size = 3
    img_with_border = cv2.copyMakeBorder(img, border_size, border_size,
                                          border_size, border_size,
                                          cv2.BORDER_CONSTANT, value=color)
    return img_with_border

def create_comparison_figure(sample_id, nir_path, census_path, cfm_path, gt_path, output_path):
    """创建单样本对比图：NIR + GT + Census + CFM"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f'跨模态立体匹配结果对比 - 样本 {sample_id}', fontsize=16, y=0.98)

    # 1. NIR 左图
    nir_img = cv2.imread(str(nir_path), cv2.IMREAD_COLOR)
    nir_img = cv2.cvtColor(nir_img, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(nir_img)
    axes[0, 0].set_title('NIR 左图 (输入)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # 2. GT 视差
    gt_color, gt_disp = load_ground_truth(gt_path)
    if gt_color is not None:
        axes[0, 1].imshow(gt_color)
        axes[0, 1].set_title('GT 视差图 (真值)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, '无GT', ha='center', va='center')
        axes[0, 1].axis('off')

    # 3. RGB 右图（输入的另一模态）
    rgb_path = str(nir_path).replace('nir/img_left', 'rgb/img_right')
    if os.path.exists(rgb_path):
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        axes[0, 2].imshow(rgb_img)
        axes[0, 2].set_title('RGB 右图 (输入)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

    # 4. Census + SGM
    census_color, census_disp = load_disparity_color(census_path)
    if census_color is not None:
        axes[1, 0].imshow(census_color)
        axes[1, 0].set_title('Census + SGM (传统方法)', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

    # 5. CFM Pipeline
    cfm_color, cfm_disp = load_disparity_color(cfm_path)
    if cfm_color is not None:
        axes[1, 1].imshow(cfm_color)
        axes[1, 1].set_title('CFM Pipeline (深度学习)', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')

    # 6. 误差热力图 (CFM vs GT)
    if gt_disp is not None and cfm_disp is not None:
        mask = gt_disp > 0
        error = np.abs(cfm_disp - gt_disp) * mask.astype(float)

        # 截断显示：误差>3px标为红色
        error_vis = np.clip(error, 0, 5)

        error_im = axes[1, 2].imshow(error_vis, cmap='hot', vmin=0, vmax=5)
        axes[1, 2].set_title('CFM 误差热力图 (|预测-GT|)', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')

        # 添加颜色条
        cbar = fig.colorbar(error_im, ax=axes[1, 2], fraction=0.046, pad=0.04)
        cbar.set_label('视差误差 (px)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'已生成: {output_path}')

def create_summary_figure(epe_census, epe_cfm, d1_census, d1_cfm, mse_census, mse_cfm, output_path):
    """创建指标对比柱状图"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('跨模态立体匹配算法性能对比', fontsize=16, y=0.98)

    methods = ['Census + SGM\n(传统方法)', 'CFM Pipeline\n(深度学习)']
    colors = ['#FF6B6B', '#4ECDC4']

    # EPE 对比
    axes[0].bar(methods, [epe_census, epe_cfm], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_title('端点误差 EPE (越低越好)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('EPE (px)')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate([epe_census, epe_cfm]):
        axes[0].text(i, v + 0.5, f'{v:.2f}', ha='center', fontweight='bold')

    # D1-all 对比
    axes[1].bar(methods, [d1_census, d1_cfm], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_title('错误匹配率 D1-all (越低越好)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('D1-all (%)')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate([d1_census, d1_cfm]):
        axes[1].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold')

    # MSE 对比 (对数刻度)
    axes[2].bar(methods, [mse_census, mse_cfm], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[2].set_title('均方误差 MSE (对数刻度)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('MSE (px², log scale)')
    axes[2].set_yscale('log')
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate([mse_census, mse_cfm]):
        axes[2].text(i, v * 1.1, f'{v:.1f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'已生成: {output_path}')

def create_overview_figure(samples, output_path):
    """创建多样本概览图"""

    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    fig.suptitle('跨模态立体匹配 - 多样本结果概览', fontsize=16, y=0.99)

    for row, sample in enumerate(samples):
        # NIR
        nir_img = cv2.imread(str(sample['nir']), cv2.IMREAD_COLOR)
        nir_img = cv2.cvtColor(nir_img, cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(nir_img)
        if row == 0:
            axes[row, 0].set_title('NIR 输入', fontsize=11, fontweight='bold')
        axes[row, 0].set_ylabel(f'样本 {sample["id"]}', fontsize=10)
        axes[row, 0].set_yticks([])
        axes[row, 0].set_xticks([])

        # GT
        gt_color, _ = load_ground_truth(sample['gt'])
        if gt_color is not None:
            axes[row, 1].imshow(gt_color)
            if row == 0:
                axes[row, 1].set_title('GT 视差', fontsize=11, fontweight='bold')
        axes[row, 1].set_yticks([])
        axes[row, 1].set_xticks([])

        # Census
        census_color, _ = load_disparity_color(sample['census'])
        if census_color is not None:
            axes[row, 2].imshow(census_color)
            if row == 0:
                axes[row, 2].set_title('Census + SGM', fontsize=11, fontweight='bold')
        axes[row, 2].set_yticks([])
        axes[row, 2].set_xticks([])

        # CFM
        cfm_color, _ = load_disparity_color(sample['cfm'])
        if cfm_color is not None:
            axes[row, 3].imshow(cfm_color)
            if row == 0:
                axes[row, 3].set_title('CFM Pipeline', fontsize=11, fontweight='bold')
        axes[row, 3].set_yticks([])
        axes[row, 3].set_xticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'已生成: {output_path}')

def main():
    base_path = Path('/home/cty/work/cv/project5/cross-modal-stereo-matching')
    output_dir = base_path / 'output' / 'figures'
    output_dir.mkdir(exist_ok=True, parents=True)

    # 样本ID列表 (取前5个)
    sample_ids = ['008000_0', '008001_1', '008002_2', '008003_3', '008004_4']

    # ========== 1. 生成单个样本对比图 ==========
    print("=" * 50)
    print("生成单个样本对比图...")
    print("=" * 50)

    for sample_id in sample_ids:
        nir_path = base_path / 'data/MS2/sync_data/nir/img_left' / f'{sample_id.split("_")[0]}.png'
        census_path = base_path / 'output/census_val' / f'{sample_id}_disp.png'
        cfm_path = base_path / 'output/cfm_full_6gb_val' / f'{sample_id}_disp.png'
        gt_path = base_path / 'data/MS2/proj_depth/nir/depth_filtered' / f'{sample_id.split("_")[0]}.png'

        if all(os.path.exists(p) for p in [nir_path, census_path, cfm_path, gt_path]):
            output_path = output_dir / f'comparison_{sample_id}.png'
            create_comparison_figure(sample_id, nir_path, census_path, cfm_path, gt_path, output_path)
        else:
            print(f"跳过样本 {sample_id}: 文件缺失")

    # ========== 2. 生成指标对比图 ==========
    print("\n" + "=" * 50)
    print("生成指标对比图...")
    print("=" * 50)

    # 评估结果
    epe_census = 56.61
    d1_census = 95.20
    mse_census = 4655.29

    epe_cfm = 14.90
    d1_cfm = 86.49
    mse_cfm = 354.57

    create_summary_figure(epe_census, epe_cfm, d1_census, d1_cfm, mse_census, mse_cfm,
                          output_dir / 'metrics_comparison.png')

    # ========== 3. 生成多样本概览图 ==========
    print("\n" + "=" * 50)
    print("生成多样本概览图...")
    print("=" * 50)

    samples = []
    for sample_id in sample_ids[:4]:  # 取前4个
        nir_path = base_path / 'data/MS2/sync_data/nir/img_left' / f'{sample_id.split("_")[0]}.png'
        census_path = base_path / 'output/census_val' / f'{sample_id}_disp.png'
        cfm_path = base_path / 'output/cfm_full_6gb_val' / f'{sample_id}_disp.png'
        gt_path = base_path / 'data/MS2/proj_depth/nir/depth_filtered' / f'{sample_id.split("_")[0]}.png'

        if all(os.path.exists(p) for p in [nir_path, census_path, cfm_path, gt_path]):
            samples.append({
                'id': sample_id,
                'nir': nir_path,
                'census': census_path,
                'cfm': cfm_path,
                'gt': gt_path
            })

    if samples:
        create_overview_figure(samples, output_dir / 'overview_all_samples.png')

    print("\n" + "=" * 50)
    print(f"所有图像已生成到: {output_dir}")
    print("=" * 50)

if __name__ == '__main__':
    main()
