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
            self.temperature = temperature
            
        def forward(self, features, labels):
            """
            Compute supervised contrastive loss via feature similarity matrix
            """
            # Inject numerical stability verification for NaN or infinite values
            if torch.isnan(features).any() or torch.isinf(features).any():
                return torch.tensor(0.0, device=features.device, requires_grad=True)
            
            batch_size = features.shape[0]
            
            # Normalize feature vectors into unit hypersphere
            features = F.normalize(features, dim=1, eps=1e-8)
            
            # Compute similarity matrix via dot product representing projection scores
            similarity_matrix = torch.matmul(features, features.T)
            similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0)
            
            # Extract fixed temperature ensuring strict positivity via manual bounds
            temperature = torch.tensor(self.temperature, device=features.device)
            
            # Prevent division by zero and temperature saturation causing NaN logits
            safe_temperature = torch.clamp(temperature, min=1e-2)
            logits = similarity_matrix / safe_temperature
            
            # Mask diagonal entries to eliminate trivial self-contrastive optimization shortcuts
            mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
            logits.masked_fill_(mask, float('-inf'))
            
            # Compute logarithmic softmax over scaled similarities
            log_probs = F.log_softmax(logits, dim=1)
            
            # Build index matrix mapping each instance to its positive class neighbors
            L_cl = []
            for i in range(batch_size):
                # Identify indices belonging to the identical class cluster
                same_class_indices = torch.where(labels == labels[i])[0]
                # Exclude the current instance itself
                same_class_indices = same_class_indices[same_class_indices != i]
                L_cl.append(same_class_indices)
            
            # Accumulate average contrastive loss across all instances
            loss = torch.tensor(0.0, device=features.device)
            for i in range(batch_size):
                if len(L_cl[i]) > 0:
                    # Gather logarithmic probabilities of all valid positive matches
                    pos_logits = log_probs[i, L_cl[i]]
                    # Aggregate sample loss via negative log likelihood
                    sample_loss = -torch.mean(pos_logits)
                    loss += sample_loss
            
            # Normalize computed total loss against valid batch size
            return loss / batch_size if batch_size > 0 else loss

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