import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import EmotionDataset
from data_processor import DataProcessor
from model import VADConfig, VADModelWithGating

# ==========================================
# 1. 线性 CKA 核心实现
# ==========================================
def linear_HSIC(X, Y):
    """
    计算线性核的 HSIC (Hilbert-Schmidt Independence Criterion)
    X: [N, D1], Y: [N, D2]
    """
    # 中心化特征
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    # 计算 tr(K_x * K_y) = ||Y^T * X||_F^2
    # 对于线性核 K = XX^T
    # HSIC = trace(XX^T YY^T) = trace(X^T Y Y^T X) = ||X^T Y||_F^2
    hsic = np.linalg.norm(X_centered.T @ Y_centered, ord='fro')**2
    return hsic

def calculate_CKA(X, Y):
    """
    计算线性 CKA 相似度
    """
    hsic_xy = linear_HSIC(X, Y)
    hsic_xx = linear_HSIC(X, X)
    hsic_yy = linear_HSIC(Y, Y)
    
    cka_score = hsic_xy / (np.sqrt(hsic_xx) * np.sqrt(hsic_yy))
    return cka_score

# ==========================================
# 2. 特征收集 (仅选取有效帧)
# ==========================================
def collect_data_for_cka(loader, model, device, max_frames=15000):
    model.eval()
    features_list = {'e2v_raw': [], 'hub_raw': [], 'e2v_proj': [], 'hub_proj': []}
    total_frames = 0
    
    print(f"正在收集特征（目标有效帧数: {max_frames}）...")
    fusion_layer = model.feature_fusion
    
    with torch.no_grad():
        for batch in loader:
            if total_frames >= max_frames: break
            
            mask = batch["padding_mask"] # [B, T]
            e2v_raw = batch["emotion2vec_features"]
            hub_raw = batch["hubert_features"]
            
            # 运行模型内部的对齐层
            e2v_tensor = e2v_raw.to(device)
            hub_tensor = hub_raw.to(device)
            
            e2v_proj = fusion_layer.feature_transforms['emotion2vec'](
                fusion_layer.feature_norms['emotion2vec'](e2v_tensor)
            )
            hub_proj = fusion_layer.feature_transforms['hubert'](
                fusion_layer.feature_norms['hubert'](hub_tensor)
            )
            
            # 过滤有效帧 (非 Padding 帧)
            valid_mask = ~mask
            
            # 提取有效数据并转为 numpy
            e2v_raw_np = e2v_raw[valid_mask].cpu().numpy()
            hub_raw_np = hub_raw[valid_mask].cpu().numpy()
            e2v_proj_np = e2v_proj[valid_mask.to(device)].cpu().numpy()
            hub_proj_np = hub_proj[valid_mask.to(device)].cpu().numpy()
            
            features_list['e2v_raw'].append(e2v_raw_np)
            features_list['hub_raw'].append(hub_raw_np)
            features_list['e2v_proj'].append(e2v_proj_np)
            features_list['hub_proj'].append(hub_proj_np)
            
            total_frames += valid_mask.sum().item()
            print(f"  - 已收集 {total_frames} 帧", end='\r')

    # 合并并截断
    final_features = {}
    for k in features_list:
        final_features[k] = np.concatenate(features_list[k], axis=0)[:max_frames].astype(np.float64)
        
    return final_features

# ==========================================
# 3. 计算矩阵并绘图
# ==========================================
def main():
    # 路径配置 (请根据实际情况调整)
    EMO2VEC_DIR = './Features/emo2vec_large_features'
    HUBERT_DIR = './Features/hubert_large_features'
    CSV_PATH = './csv_files/IEMOCAP.csv'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备数据集
    dataset = EmotionDataset(EMO2VEC_DIR, HUBERT_DIR, CSV_PATH)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=DataProcessor.collate_fn)
    
    # 初始化模型结构
    config = VADConfig(emotion2vec_dim=1024, hubert_dim=1024)
    model = VADModelWithGating(config).to(device)
    
    # 1. 收集数据 (取 15000 帧左右即可保证统计显著性)
    features = collect_data_for_cka(loader, model, device, max_frames=15000)
    
    keys = ['e2v_raw', 'hub_raw', 'e2v_proj', 'hub_proj']
    labels = ['E2V (Raw)', 'Hub (Raw)', 'E2V (Proj)', 'Hub (Proj)']
    
    # 2. 计算 CKA 矩阵
    n = len(keys)
    cka_matrix = np.zeros((n, n))
    print("\n\n正在计算 CKA 相似度矩阵 (Linear Kernels)...")
    
    for i in range(n):
        for j in range(i, n): # 矩阵是对称的
            score = calculate_CKA(features[keys[i]], features[keys[j]])
            cka_matrix[i, j] = score
            cka_matrix[j, i] = score
            print(f"  - [{labels[i]}] vs [{labels[j]}] Similarity: {score:.4f}")

    # 3. 绘图 (论文发表风格优化)
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white", context="paper", font_scale=1.4)
    
    # 绘制热力图
    mask = np.triu(np.ones_like(cka_matrix, dtype=bool), k=1) # 可选：只画下三角或全画
    sns.heatmap(cka_matrix, annot=True, fmt=".3f", cmap="Blues", 
                xticklabels=labels, yticklabels=labels, square=True,
                linewidths=1, cbar_kws={"shrink": .8, "label": "CKA Score"})
    
    plt.title("CKA Similarity Matrix: Raw vs. Projected Representations", fontsize=16, pad=25)
    plt.tight_layout()
    
    # 保存结果
    output_path = './AnalysisResults/cka_similarity_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n分析成功！CKA 矩阵已保存至 '{output_path}'")
    plt.show()

if __name__ == "__main__":
    main()
