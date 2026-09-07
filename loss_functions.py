"""
Loss function module
Contains diverse loss function implementations utilized during neural network training
"""

import torch
from torch import nn
import torch.nn.functional as F

class LossFactory:
    """
    Loss factory class instantiating various training loss functions
    """
    class SupervisedContrastiveLoss(nn.Module):
        """
        Supervised contrastive loss implementation
        """
        def __init__(self, temperature=0.1):
            """
            Initialize supervised contrastive loss with fixed scalar temperature
            """
            super().__init__()
            if temperature <= 0:
                raise ValueError(f"temperature must be positive, but got {temperature}")
            self.temperature = float(temperature)
            
        def forward(self, features, labels):
            """
            Compute supervised contrastive loss via feature similarity matrix
            """
            if features.ndim != 2:
                raise ValueError(
                    f"features must have shape [B, D], but got {tuple(features.shape)}"
                )
            if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
                raise ValueError(
                    "labels must have shape [B] and match the feature batch size"
                )
            if not torch.isfinite(features).all():
                raise ValueError("Supervised contrastive features contain NaN or Inf")
            
            batch_size = features.shape[0]
            
            # Normalize feature vectors into unit hypersphere
            features = F.normalize(features, dim=1, eps=1e-8)
            
            # Compute similarity matrix via dot product representing projection scores
            similarity_matrix = torch.matmul(features, features.T)
            logits = similarity_matrix / self.temperature
            
            # Mask diagonal entries to eliminate trivial self-contrastive optimization shortcuts
            mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
            logits.masked_fill_(mask, float('-inf'))
            
            # Compute logarithmic softmax over scaled similarities
            log_probs = F.log_softmax(logits, dim=1)
            
            # Positive pairs share a class but exclude the anchor itself.
            positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~mask
            positive_counts = positive_mask.sum(dim=1)
            valid_anchors = positive_counts > 0

            # A batch with no positive pair contributes a graph-connected zero,
            # so joint-loss backward remains valid.
            if not valid_anchors.any():
                return features.sum() * 0.0

            selected_log_probs = torch.where(
                positive_mask,
                log_probs,
                torch.zeros_like(log_probs),
            )
            mean_positive_log_prob = (
                selected_log_probs.sum(dim=1)
                / positive_counts.clamp_min(1)
            )

            # Average over anchors for which P(i) is non-empty.
            return -mean_positive_log_prob[valid_anchors].mean()

    class CCCLoss(nn.Module):
        """
        Concordance Correlation Coefficient loss tailored for continuous regression problems
        """
        def __init__(self):
            """
            Initialize concordance correlation coefficient objective
            """
            super().__init__()
            
        def forward(self, preds, labels):
            """
            Compute CCC loss aggregating over valence arousal and dominance axes
            """
            ccc_v = LossFactory._compute_dimension_ccc(preds[:, 0], labels[:, 0])
            ccc_a = LossFactory._compute_dimension_ccc(preds[:, 1], labels[:, 1])
            ccc_d = LossFactory._compute_dimension_ccc(preds[:, 2], labels[:, 2])
            
            mean_ccc = (ccc_v + ccc_a + ccc_d) / 3.0
            return torch.tensor(1.0, device=preds.device) - mean_ccc

    @staticmethod
    def _compute_dimension_ccc(preds, labels):
        """
        Calculate scalar Concordance Correlation Coefficient for a single psychological dimension
        """
        preds_mean = torch.mean(preds)
        labels_mean = torch.mean(labels)
        
        preds_var = torch.mean((preds - preds_mean) ** 2)
        labels_var = torch.mean((labels - labels_mean) ** 2)
        
        covar = torch.mean((preds - preds_mean) * (labels - labels_mean))
        
        ccc = 2 * covar / (preds_var + labels_var + (preds_mean - labels_mean) ** 2 + 1e-8)
        return ccc
