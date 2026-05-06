"""
Dataset module
Handles loading and processing of affective data
"""

import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import ast

class EmotionDataset(torch.utils.data.Dataset):
    """
    Emotion dataset class for managing affective data samples
    """
    def __init__(self, emotion2vec_dir, hubert_dir, csv_path, wav2vec_dir=None, data2vec_dir=None):
        """
        Initialize dataset with feature directories and label annotations
        """
        self.df = pd.read_csv(csv_path)
        self.emotion2vec_dir = emotion2vec_dir
        self.hubert_dir = hubert_dir
        self.wav2vec_dir = wav2vec_dir
        self.data2vec_dir = data2vec_dir
 
        # Map categorical emotion labels to numerical indices
        self.emotion_map = {
            'neu': 0, 
            'hap': 1,
            'ang': 2,
            'sad': 3,
            'sur': 4,
            'fea': 5,
            'dis': 6,
            'con': 7
        }

        self.vad_labels = []
        for vad_str in self.df['VAD_normalized']:
            # Use safe evaluation for parsed list structures
            vad = ast.literal_eval(vad_str)
            self.vad_labels.append(torch.tensor(vad, dtype=torch.float))
        
        self.emotion_labels = []
        for label in self.df['Label']:
            # Convert categorical labels to one-hot encoding vectors
            label_idx = self.emotion_map[label]
            one_hot = torch.zeros(len(self.emotion_map))
            one_hot[label_idx] = 1
            self.emotion_labels.append(one_hot)

    def __len__(self):
        """
        Return the total number of samples in the dataset
        """
        return len(self.df)
        
    def __getitem__(self, idx):
        """
        Retrieve and process a single data sample
        """
        row = self.df.iloc[idx]
        base_filename = os.path.splitext(row['FileName'])[0]
        
        # Load essential feature modalities
        emotion2vec_path = os.path.join(self.emotion2vec_dir, f"{base_filename}.npy")
        hubert_path = os.path.join(self.hubert_dir, f"{base_filename}.npy")
        
        emotion2vec_features = torch.from_numpy(np.load(emotion2vec_path)).float()
        hubert_features = torch.from_numpy(np.load(hubert_path)).float()
        
        # Load supplementary feature modalities if available
        wav2vec_features = None
        data2vec_features = None
        
        if self.wav2vec_dir:
            wav2vec_path = os.path.join(self.wav2vec_dir, f"{base_filename}.npy")
            wav2vec_features = torch.from_numpy(np.load(wav2vec_path)).float()
            
        if self.data2vec_dir:
            data2vec_path = os.path.join(self.data2vec_dir, f"{base_filename}.npy")
            data2vec_features = torch.from_numpy(np.load(data2vec_path)).float()
        
        # Aggregate all loaded features
        all_features = [emotion2vec_features, hubert_features]
        if wav2vec_features is not None:
            all_features.append(wav2vec_features)
        if data2vec_features is not None:
            all_features.append(data2vec_features)
        
        target_len = max([feat.size(0) for feat in all_features])
        
        # Align all features to the maximum sequence length via linear interpolation
        for i in range(len(all_features)):
            if all_features[i].size(0) != target_len:
                # Interpolate temporal sequence length along the correct dimension
                x = all_features[i].transpose(0, 1).unsqueeze(0)
                all_features[i] = F.interpolate(
                    x,
                    size=target_len,
                    mode='linear',
                    align_corners=True
                ).squeeze(0).transpose(0, 1)
        
        result = {
            "id": row['FileName'],
            "emotion2vec_features": all_features[0],
            "hubert_features": all_features[1],
            "labels": self.vad_labels[idx],
            "emotion_labels": self.emotion_labels[idx]
        }
        
        # Inject supplementary features into the output dictionary
        if wav2vec_features is not None:
            result["wav2vec_features"] = all_features[2]
        if data2vec_features is not None:
            result["data2vec_features"] = all_features[3 if wav2vec_features is not None else 2]
        
        return result
