"""
Training and evaluation module
Orchestrates neural network training iterations and performance validation
"""

import torch
from tqdm import tqdm
from loss_functions import LossFactory

class TrainingManager:
    """
    Training manager encapsulating functions for training epochs and model validation
    """
    @staticmethod
    def train_one_epoch(model, optimizer, vad_criterion, contrast_criterion, train_loader, device):
        """
        Execute a single training epoch over the entire dataset
        """
        model.train()
        total_loss = 0.0
        total_batches = 0
        total_weight_sums = {}
        total_weight_counts = {}
        total_vad_loss = 0.0
        total_contrast_loss = 0.0
        
        optimizer.zero_grad()
        
        progress_bar = tqdm(total=len(train_loader), desc='Training', leave=False)
        
        for batch_idx, batch in enumerate(train_loader):

            optimizer.zero_grad()

            # Prepare feature dictionary
            features = {
                "emotion2vec": batch["emotion2vec_features"].to(device),
                "hubert": batch["hubert_features"].to(device)
            }
            
            # Incorporate supplementary features if available
            if "wav2vec_features" in batch:
                features["wav2vec"] = batch["wav2vec_features"].to(device)
            if "data2vec_features" in batch:
                features["data2vec"] = batch["data2vec_features"].to(device)

            padding_mask = batch["padding_mask"].to(device)
            vad_labels = batch["labels"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            emotion_indices = torch.argmax(emotion_labels, dim=1)
            
            vad_preds, feature_weights, contrast_features, current_temp = model(
                features,
                padding_mask
            )
            
            vad_loss = vad_criterion(vad_preds, vad_labels)
            contrast_loss = contrast_criterion(contrast_features, emotion_indices)
            loss = 1.0 * vad_loss + 0.6 * contrast_loss

            # Backpropagation and parameter updates
            loss.backward()
            
            # Gradient clipping must be executed after backward and before optimizer step to affect all parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            # Explicit cache clearing is removed to prevent severe training slowdowns

            # Record metrics
            total_loss += loss.item()
            total_vad_loss += vad_loss.item()
            total_contrast_loss += contrast_loss.item()
            total_batches += 1
            
            # Record group-wise gating weights using valid frames only.
            valid_frames = (~padding_mask).unsqueeze(-1)
            for feat_type, weight in feature_weights.items():
                if weight.shape[:2] != padding_mask.shape:
                    raise ValueError(
                        f"Gate weights for {feat_type} have incompatible shape "
                        f"{tuple(weight.shape)}"
                    )

                expanded_valid_frames = valid_frames.expand_as(weight)
                valid_weights = weight.masked_select(expanded_valid_frames)

                if feat_type not in total_weight_sums:
                    total_weight_sums[feat_type] = 0.0
                    total_weight_counts[feat_type] = 0
                total_weight_sums[feat_type] += valid_weights.sum().item()
                total_weight_counts[feat_type] += valid_weights.numel()

            # Update progress tracking
            avg_weights = {
                f"{k}_w": f"{total_weight_sums[k] / total_weight_counts[k]:.3f}"
                for k in total_weight_sums
                if total_weight_counts[k] > 0
            }
            postfix_info = {'loss': f'{(total_loss/(batch_idx+1)):.4f}'}
            postfix_info.update(avg_weights)
            progress_bar.set_postfix(postfix_info)
            progress_bar.update(1)
        
        progress_bar.close()
        
        # Return epoch results
        result = {
            'loss': total_loss / len(train_loader),
            'vad_loss': total_vad_loss / len(train_loader),
            'contrast_loss': total_contrast_loss / len(train_loader),
        }
    
        # Append average weights for each feature modality
        for feat_type in total_weight_sums:
            result[f'{feat_type}_weight'] = (
                total_weight_sums[feat_type] / total_weight_counts[feat_type]
            )
        
        return result

    @staticmethod
    def validate_and_test(model, data_loader, device):
        """
        Validate and test model generalization performance
        """
        model.eval()
        all_vad_preds = []
        all_vad_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating', leave=False):
                # Prepare feature dictionary
                features = {
                    "emotion2vec": batch["emotion2vec_features"].to(device),
                    "hubert": batch["hubert_features"].to(device)
                }
                
                # Incorporate supplementary features if available
                if "wav2vec_features" in batch:
                    features["wav2vec"] = batch["wav2vec_features"].to(device)
                if "data2vec_features" in batch:
                    features["data2vec"] = batch["data2vec_features"].to(device)
                    
                padding_mask = batch["padding_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Resolve return value confusion by correctly identifying gate weights instead of discrete logits
                vad_preds, gate_weights, _, _ = model(features, padding_mask)
                
                all_vad_preds.append(vad_preds)
                all_vad_labels.append(batch["labels"].to(device))
                
        
        # Concatenate evaluation batches
        all_vad_preds = torch.cat(all_vad_preds, dim=0)
        all_vad_labels = torch.cat(all_vad_labels, dim=0)
        
        # Compute concordance correlation coefficient metrics
        ccc_v = LossFactory._compute_dimension_ccc(all_vad_preds[:, 0], all_vad_labels[:, 0]).item()
        ccc_a = LossFactory._compute_dimension_ccc(all_vad_preds[:, 1], all_vad_labels[:, 1]).item()
        ccc_d = LossFactory._compute_dimension_ccc(all_vad_preds[:, 2], all_vad_labels[:, 2]).item()
        
        return ccc_v, ccc_a, ccc_d
