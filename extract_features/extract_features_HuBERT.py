import os
import torch
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel

def batch_extract_and_save(input_dir, output_dir):
    # Prepare directories and file lists
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define supported audio extensions
    valid_extensions = {'.wav', '.mp3', '.flac'}
    audio_files = [f for f in input_path.rglob('*') if f.suffix.lower() in valid_extensions]

    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return

    # Initialize model and computing device
    print("Loading HuBERT large model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/hubert-large-ll60k"
    
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name).to(device)
    model.eval()

    # Iterate and extract features
    print(f"Starting batch extraction for {len(audio_files)} files on device {device}")
    
    for audio_file in tqdm(audio_files, desc="Feature extraction progress"):
        # Construct output file path
        output_file = output_path / f"{audio_file.stem}.npy"
        
        # Skip if file already exists
        if output_file.exists():
            continue

        try:
            # Read audio data
            speech, sr = librosa.load(audio_file, sr=16000)
            
            # Transfer data to target computing device
            inputs = processor(speech, return_tensors="pt", sampling_rate=16000).to(device)

            # Extract hierarchical features
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Fuse hidden states from layer 10 to 15
            target_layers = outputs.hidden_states[10:16]
            fused_features = torch.mean(torch.stack(target_layers), dim=0).squeeze(0)

            # Convert to float16 numpy array and save
            features_np = fused_features.cpu().numpy().astype('float16')
            np.save(output_file, features_np)

        except Exception as e:
            print(f"\nError processing {audio_file.name}: {str(e)}")

    print("\nFeature processing complete")

# Execute extraction pipeline
if __name__ == "__main__":
    INPUT_DIRECTORY = "/home/victor/DataSet/IEMOCAP"   
    OUTPUT_DIRECTORY = "/home/victor/Github/DGFF-SER/hubert_large_features" 
    
    batch_extract_and_save(INPUT_DIRECTORY, OUTPUT_DIRECTORY)