# DGFF-SER: Dynamic Gated Feature Fusion for Speech Emotion Recognition

## 项目简介

DGFF-SER 是一个基于深度学习的高级情感语音分析系统，旨在预测语音在 Valence（愉悦度）、Arousal（激活度）和 Dominance（支配度）三个连续情感上的数值。

*   **多特征融合**：支持同时输入 Emotion2Vec、HuBERT 以及 Wav2Vec、Data2Vec（可选）等预训练模型提取的高维语音表征。
*   **多粒度门控机制 (Multi-Grained Gating)**：将特征分割为多个独立的组，使用固定温度系数（默认 0.1）的 Softmax 门控网络自适应地计算各特征来源的重要性权重，在每个组内进行加权求和，最后结合双向 LSTM 捕获长程时序依赖。
*   **双重目标优化**：
    *   **一致性相关系数损失 (CCC Loss)**：专门针对情感回归任务优化的指标级损失。
    *   **监督对比学习 (Supervised Contrastive Learning)**：通过非线性投影头，利用离散情感标签拉近同类样本的潜在空间距离，增强模型表征能力。

---

## 目录结构

```text
.
├── baseline.py                 # 基线模型实现
├── data_processor.py           # 批次数据处理与特征序列零填充对齐
├── dataset.py                  # 自定义情感数据集类及安全的标签解析
├── early_stopping.py           # 基于验证集性能的早停防过拟合机制
├── loss_functions.py           # CCC 损失与严谨的监督对比损失实现
├── lr_scheduler.py             # 学习率调度策略
├── main.py                     # 全局配置、参数解析与主程序入口
├── model_components.py         # 多粒度门控、时序网络与注意力池化等底层组件
├── model.py                    # VAD 主模型定义、投影头与前向传播逻辑
├── trainer_executor.py         # 统筹 Fold 级别生命周期、优化器与日志管理
├── trainer.py                  # 执行单轮训练与测试，实施精准的梯度裁剪
├── util.py                     # 提供不同数据集的动态划分算法
├── UMAP.py						# 高维特征降维与可视化工具
├── config/                     # 项目配置文件目录
├── csv_files/                  # 存储各数据集的标注与元数据 CSV
├── extract_features/           # 调用各种预训练大模型提取底层特征的脚本集
└── scripts/                    # 批处理 Shell 脚本目录
```

---

## 核心模块说明

*   `main.py`: 程序的唯一入口。负责解析命令行参数、锁定全局随机种子以确保深度学习实验的严谨可复现性，并根据传入的 CSV 文件名动态选择数据集划分策略（例如自动识别 MSP-Podcast 或 IEMOCAP）。
*   `dataset.py`: 提供 `EmotionDataset` 类。负责加载和配对多种 `.npy` 特征文件，利用 `ast.literal_eval` 安全解析字符串格式的 VAD 标签，并将不同采样率的特征通过严格的插值算法对齐到统一序列长度。
*   `data_processor.py`: 提供 DataLoader 使用的 `collate_fn`。负责在组装 Batch 时动态计算最大长度，运用零填充将序列整理为规则张量，并生成对应的 Padding Mask。
*   `model.py` & `model_components.py`: 模型主体。定义了带有多层 Transformer 编码器的 `VADModelWithGating`，以及采用固定温度 Softmax 和组内加权求和的多粒度特征融合网络、注意力池化层和对比学习专属投影头。
*   `trainer.py` & `trainer_executor.py`: 模型训练。`trainer_executor.py` 管理的 epoch 循环、学习率步进与最佳模型保存；`trainer.py` 专注于微观计算，在反向传播与优化器更新之间实施关键的梯度裁剪，并计算损失。
*   `loss_functions.py`: 提供计算相关性指标的 `CCCLoss。

---

## 环境要求

本项目在Ubuntu24.4以及 Python 3.12+ 环境进行测试。
---

## 运行项目

### 1. 特征提取
在训练前，需要使用 `extract_features/` 目录下的脚本为你的音频提取预训练特征：
```bash
# 示例：提取 HuBERT 特征
python extract_features/extract_features_HuBERT.py
```

### 2. 开始训练
使用 `main.py` 启动训练流水线。提供 Emotion2Vec 和 HuBERT 特征目录以及对应的标注 CSV 文件。程序会基于你提供的随机种子 (`--seed`) 保证结果完全一致。

**基本训练命令：**
```bash
python main.py \
    --emotion2vec_dir /path/to/emo2vec_features \
    --hubert_dir /path/to/hubert_features \
    --csv_path ./csv_files/IEMOCAP.csv \
    --save_dir ./models/run_001 \
    --batch_size 16 \
    --epochs 50 \
    --lr 2e-5 \
    --seed 42 \
    --patience 10
```

**融合更多特征：**
如果提取了 Wav2Vec 或 Data2Vec 特征，可以直接通过参数加入融合网络：
```bash
python main.py \
    --emotion2vec_dir /path/to/emo2vec_features \
    --hubert_dir /path/to/hubert_features \
    --wav2vec_dir /path/to/wav2vec_features \
    --csv_path ./csv_files/MSP_Podcast.csv \
    --save_dir ./models/run_002
```

### 3. 输出与监控
*   **日志记录**：训练过程中的 loss 变化、各特征的动态门控权重（例如 `emotion2vec_w`, `hubert_w`）以及验证集的 CCC 指标会实时输出在终端，并持久化到 `--save_dir` 中的 `training.log` 和 `metrics.csv`。
*   **模型保存**：触发 Early Stopping 或训练结束后，性能最优的模型权重将保存在 `best_model/` 目录下。
*   **结果分析**：最终的交叉验证平均 CCC 得分或独立测试集得分会保存在 `final_results.txt` 中。