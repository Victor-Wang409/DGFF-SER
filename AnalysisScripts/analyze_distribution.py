import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import EmotionDataset
from data_processor import DataProcessor
from model import VADConfig, VADModelWithGating

# ==========================================
# 1. 环境初始化
# ==========================================
def setup_analysis():
    class Args:
        emotion2vec_dir = './emo2vec_large_features'
        hubert_dir = './hubert_large_features'
        csv_path = './csv_files/IEMOCAP.csv'
        batch_size = 8 # 稍微减小 Batch 以稳定内存
    
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = EmotionDataset(args.emotion2vec_dir, args.hubert_dir, args.csv_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=DataProcessor.collate_fn)
    
    config = VADConfig(emotion2vec_dim=1024, hubert_dim=1024)
    model = VADModelWithGating(config).to(device)
    model.eval()
    
    return loader, model, device

# ==========================================
# 2. 特征收集 (优化版: 剔除 Padding 且 子采样)
# ==========================================
def collect_distributions(loader, model, device, num_batches=10, points_limit=200000):
    data = {
        'raw_e2v': [], 'raw_hub': [],
        'proj_e2v': [], 'proj_hub': []
    }
    
    print(f"正在分析特征分布（剔除 Padding 并限制单模态最大采样点为 {points_limit}）...")
    fusion_layer = model.feature_fusion
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches: break
            
            # 获取 Padding Mask (False 表示有效, True 表示补零)
            mask = batch["padding_mask"] # [B, T]
            
            # A. 处理原始特征 (Raw)
            # 使用 mask 选取非 Padding 帧
            e2v_raw = batch["emotion2vec_features"][~mask].cpu().numpy().flatten()
            hub_raw = batch["hubert_features"][~mask].cpu().numpy().flatten()
            
            data['raw_e2v'].extend(e2v_raw)
            data['raw_hub'].extend(hub_raw)
            
            # B. 处理投影特征 (Projected)
            e2v_tensor = batch["emotion2vec_features"].to(device)
            hub_tensor = batch["hubert_features"].to(device)
            
            e2v_proj = fusion_layer.feature_transforms['emotion2vec'](
                fusion_layer.feature_norms['emotion2vec'](e2v_tensor)
            )
            hub_proj = fusion_layer.feature_transforms['hubert'](
                fusion_layer.feature_norms['hubert'](hub_tensor)
            )
            
            # 同样使用 mask 过滤投影后的特征
            e2v_proj_filtered = e2v_proj[~mask.to(device)].cpu().numpy().flatten()
            hub_proj_filtered = hub_proj[~mask.to(device)].cpu().numpy().flatten()
            
            data['proj_e2v'].extend(e2v_proj_filtered)
            data['proj_hub'].extend(hub_proj_filtered)
            
            print(f"  - Batch [{i+1}/{num_batches}] 完成")
            
    # 执行子采样防止绘图时内存爆炸
    for k in data:
        arr = np.array(data[k], dtype=np.float32)
        if len(arr) > points_limit:
            idx = np.random.choice(len(arr), points_limit, replace=False)
            data[k] = arr[idx]
        else:
            data[k] = arr
            
    return data

# ==========================================
# 3. 绘图 (论文发表级优化版)
# ==========================================
def plot_kde(data):
    # 设置论文风格的绘图上下文
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    # 使用更专业的调色板
    palette = {"E2V": "#4C72B0", "Hub": "#DD8452"}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- 图 1: 原始分布 (Raw) ---
    # 计算合理显示范围：取 0.5% 到 99.5% 分位数，彻底解决“尖峰”导致的视觉压缩问题
    all_raw = np.concatenate([data['raw_e2v'], data['raw_hub']])
    x_min, x_max = np.percentile(all_raw, [0.5, 99.5])
    margin = (x_max - x_min) * 0.1 # 留出 10% 的边距
    
    sns.kdeplot(data['raw_e2v'], ax=axes[0], fill=True, label='Emotion2Vec (Raw)', 
                color=palette["E2V"], alpha=0.3, linewidth=2)
    sns.kdeplot(data['raw_hub'], ax=axes[0], fill=True, label='HuBERT (Raw)', 
                color=palette["Hub"], alpha=0.3, linewidth=2)
    
    axes[0].set_xlim(x_min - margin, x_max + margin)
    axes[0].set_title("1. Raw Feature Distribution\n(Outliers Removed for Visualization)", fontsize=14, pad=15)
    axes[0].set_xlabel("Feature Value Range", fontsize=12)
    axes[0].set_ylabel("Probability Density", fontsize=12)
    axes[0].legend(frameon=True, loc='upper right')

    # --- 图 2: 投影分布 (Projected) ---
    sns.kdeplot(data['proj_e2v'], ax=axes[1], fill=True, label='Emotion2Vec (Projected)', 
                color=palette["E2V"], alpha=0.3, linewidth=2)
    sns.kdeplot(data['proj_hub'], ax=axes[1], fill=True, label='HuBERT (Projected)', 
                color=palette["Hub"], alpha=0.3, linewidth=2)
    
    axes[1].set_title("2. Projected Feature Distribution\n(Normalized & Aligned Space)", fontsize=14, pad=15)
    axes[1].set_xlabel("Aligned Hidden Space Value", fontsize=12)
    axes[1].set_ylabel("Probability Density", fontsize=12)
    axes[1].legend(frameon=True, loc='upper right')

    plt.tight_layout()
    # 保存高质量图片
    output_name = 'feature_distribution_paper.png'
    plt.savefig(output_name, dpi=600, bbox_inches='tight')
    print(f"分析成功！高质量图片已保存至 '{output_name}'")
    plt.show()

if __name__ == "__main__":
    loader, model, device = setup_analysis()
    dist_data = collect_distributions(loader, model, device)
    plot_kde(dist_data)
