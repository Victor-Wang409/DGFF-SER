"""
Data processing module
Handles collation and formatting of data batches for computational efficiency
"""

import torch

class DataProcessor:
    """
    Data processor managing batch assembly and tensor formatting
    """
    @staticmethod
    def collate_fn(batch):
        """
        Data batching function unifying temporal sequences into dense rectangular tensors via zero padding
        """
        max_len = max([b["emotion2vec_features"].shape[0] for b in batch])
        
        batch_features = {}
        batch_features["emotion2vec_features"] = []
        batch_features["hubert_features"] = []
        
        # Verify presence of supplementary speech foundation representations
        has_wav2vec = "wav2vec_features" in batch[0]
        has_data2vec = "data2vec_features" in batch[0]
        
        if has_wav2vec:
            batch_features["wav2vec_features"] = []
        if has_data2vec:
            batch_features["data2vec_features"] = []
        
        batch_padding_masks = []
        batch_ids = []
        batch_labels = []
        batch_emotion_labels = []
        
        for item in batch:
            curr_len = item["emotion2vec_features"].shape[0]
            pad_len = max_len - curr_len
            
            if pad_len > 0:
                # Apply zero padding to align heterogeneous temporal lengths within a batch
                batch_features["emotion2vec_features"].append(
                    torch.cat([item["emotion2vec_features"], 
                            torch.zeros(pad_len, item["emotion2vec_features"].shape[1])], dim=0)
                )
                
                batch_features["hubert_features"].append(
                    torch.cat([item["hubert_features"], 
                            torch.zeros(pad_len, item["hubert_features"].shape[1])], dim=0)
                )
                
                if has_wav2vec:
                    batch_features["wav2vec_features"].append(
                        torch.cat([item["wav2vec_features"], 
                                torch.zeros(pad_len, item["wav2vec_features"].shape[1])], dim=0)
                    )
                    
                if has_data2vec:
                    batch_features["data2vec_features"].append(
                        torch.cat([item["data2vec_features"], 
                                torch.zeros(pad_len, item["data2vec_features"].shape[1])], dim=0)
                    )
                    
                padding_mask = torch.cat([
                    torch.zeros(curr_len),
                    torch.ones(pad_len)
                ], dim=0)
            else:
                batch_features["emotion2vec_features"].append(item["emotion2vec_features"])
                batch_features["hubert_features"].append(item["hubert_features"])
                
                if has_wav2vec:
                    batch_features["wav2vec_features"].append(item["wav2vec_features"])
                if has_data2vec:
                    batch_features["data2vec_features"].append(item["data2vec_features"])
                    
                padding_mask = torch.zeros(curr_len)
            
            batch_padding_masks.append(padding_mask)
            batch_ids.append(item["id"])
            batch_labels.append(item["labels"])
            batch_emotion_labels.append(item["emotion_labels"])
        
        # Convert batched lists into optimized multidimensional tensors
        result = {
            "id": batch_ids,
            "padding_mask": torch.stack(batch_padding_masks).bool(),
            "labels": torch.stack(batch_labels),
            "emotion_labels": torch.stack(batch_emotion_labels)
        }
        
        # Assimilate all dynamically extracted feature types
        for feat_type, feat_list in batch_features.items():
            result[feat_type] = torch.stack(feat_list)
        
        return result