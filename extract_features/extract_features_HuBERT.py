import os
import torch
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel

def batch_extract_and_save(input_dir, output_dir):
    # 1. 准备目录与文件列表
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True) # 如果输出目录不存在则创建

    # 支持的音频格式，可自行添加
    valid_extensions = {'.wav', '.mp3', '.flac'}
    audio_files = [f for f in input_path.rglob('*') if f.suffix.lower() in valid_extensions]

    if not audio_files:
        print(f"在 {input_dir} 下未找到音频文件。")
        return

    # 2. 初始化模型与设备 (移出循环，避免重复加载)
    print("正在加载 HuBERT-large 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/hubert-large-ll60k"
    
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name).to(device)
    model.eval()

    # 3. 遍历提取
    print(f"开始批量提取，共计 {len(audio_files)} 个文件。使用设备: {device}")
    
    for audio_file in tqdm(audio_files, desc="特征提取进度"):
        # 构建输出文件名 (同名，后缀改为 .pt)
        output_file = output_path / f"{audio_file.stem}.pt"
        
        # 如果文件已存在则跳过 (支持断点续传)
        if output_file.exists():
            continue

        try:
            # 读取音频
            speech, sr = librosa.load(audio_file, sr=16000)
            
            # 数据需移动到对应设备 (GPU/CPU)
            inputs = processor(speech, return_tensors="pt", sampling_rate=16000).to(device)

            # 提取特征
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # 融合第 10 到 15 层
            target_layers = outputs.hidden_states[10:16]
            fused_features = torch.mean(torch.stack(target_layers), dim=0).squeeze(0)

            # 转为 float16，移回 CPU 并保存
            torch.save(fused_features.half().cpu(), output_file)

        except Exception as e:
            print(f"\n[错误] 处理 {audio_file.name} 失败: {str(e)}")

    print("\n所有特征处理完毕！")

# 执行代码
if __name__ == "__main__":
    # 请替换为你的实际文件夹路径
    INPUT_DIRECTORY = "/home/victor/DataSet/IEMOCAP"   
    OUTPUT_DIRECTORY = "/home/victor/Github/DGFF-SER/hubert_large_features" 
    
    batch_extract_and_save(INPUT_DIRECTORY, OUTPUT_DIRECTORY)