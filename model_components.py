"""
Model components module
Contains neural architecture components enhancing feature gating and fusion
"""

import torch
from torch import nn
import torch.nn.functional as F

class ModelComponents:
    """
    Model components namespace housing building blocks for the overarching architecture
    """
    class AttentionPooling(nn.Module):
        """
        Attention pooling computing global representations from sequential features
        """
        def __init__(self, hidden_dim):
            """
            Initialize attention pooling layer
            """
            super().__init__()
            self.attention = nn.Linear(hidden_dim, 1)

        def forward(self, x, padding_mask=None):
            """
            Compute attention weights and aggregate temporal features
            """
            # Calculate attention scores
            attn_weights = self.attention(x)
            attn_weights = attn_weights.squeeze(-1)

            if padding_mask is not None:
                attn_weights = attn_weights.masked_fill(padding_mask, float('-inf'))
            attn_weights = torch.softmax(attn_weights, dim=1)
            weights_sum = torch.bmm(attn_weights.unsqueeze(1), x)
            weights_sum = weights_sum.squeeze(1)

            return weights_sum

    class MultiHeadAttention(nn.Module):
        """
        Multihead attention mechanism
        """
        def __init__(self, config):
            """
            Initialize multihead attention dimensions and projections
            """
            super().__init__()
            self.num_heads = config.num_attention_heads
            self.hidden_dim = config.hidden_dim
            self.head_dim = config.hidden_dim // config.num_attention_heads
            self.scaling = self.head_dim ** -0.5

            self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            
        def forward(self, x, key_padding_mask=None):
            """
            Execute attention computation across multiple heads
            """
            batch_size, seq_len, embed_dim = x.shape
            
            # Project inputs to queries keys and values
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Compute raw attention scores
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                )

            attn_weights = torch.softmax(attn_weights, dim=-1)
            
            # Apply attention weights to values
            attn = torch.matmul(attn_weights, v)
            
            # Reshape and compute final projection
            attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
            attn = self.out_proj(attn)
            
            return attn

    class TransformerEncoderLayer(nn.Module):
        """
        Transformer encoder layer block
        """
        def __init__(self, config):
            """
            Initialize transformer encoder layer subcomponents
            """
            super().__init__()
            self.self_attn = ModelComponents.MultiHeadAttention(config)
            
            self.linear1 = nn.Linear(config.hidden_dim, config.intermediate_dim)
            self.linear2 = nn.Linear(config.intermediate_dim, config.hidden_dim)
            
            self.norm1 = nn.LayerNorm(config.hidden_dim)
            self.norm2 = nn.LayerNorm(config.hidden_dim)
            
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.activation = nn.GELU()
            
        def forward(self, x, padding_mask=None):
            """
            Forward pass through self attention and feedforward networks
            """
            # Self attention block with residual connection
            residual = x
            x = self.norm1(x)
            x = self.self_attn(x, padding_mask)
            x = self.dropout(x)
            x = residual + x
            
            # Feedforward network block with residual connection
            residual = x
            x = self.norm2(x)
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            x = self.dropout(x)
            x = residual + x
            
            return x

    class GatedFeatureFusion(nn.Module):
        """
        Gated feature fusion mechanism
        Transforms multi grained and temporal processing into a serial pipeline
        """
        def __init__(self, feature_dims, num_groups=16, dropout_rate=0.1):
            """
            Initialize gated feature fusion
            """
            super().__init__()
            self.feature_types = list(feature_dims.keys())
            self.feature_dims = feature_dims
            self.num_features = len(self.feature_types)
            self.dropout_rate = dropout_rate
            # Initialize fixed scalar temperature for gating softmax
            self.temperature = 0.1
            # Execute weight initialization
            self._init_weights()
            
            # Validate feature count criteria
            assert 2 <= self.num_features <= 4, f"Feature count must be between 2 and 4 but got {self.num_features}"
            
            # Feature transformation and normalization modules
            self.feature_transforms = nn.ModuleDict()
            self.feature_norms = nn.ModuleDict()
            
            # Adopt first feature dimension as standard baseline
            self.standard_dim = list(feature_dims.values())[0]
            
            # Create transformation and normalization layers for each feature
            for feat_type, dim in feature_dims.items():
                # Add dropout to enhance generalization capabilities
                self.feature_transforms[feat_type] = nn.Sequential(
                    nn.Linear(dim, self.standard_dim),
                    nn.Dropout(dropout_rate)
                )
                self.feature_norms[feat_type] = nn.LayerNorm(dim)
            
            # Multi grained gating parameters
            self.num_groups = num_groups
            self.group_size = self.standard_dim // self.num_groups
            
            # Create independent gating networks for each feature group incorporating GELU
            self.group_gate_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.group_size * self.num_features, self.group_size),
                    nn.LayerNorm(self.group_size),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(self.group_size, self.num_features),
                ) for _ in range(self.num_groups)
            ])
            
            # Temporal processing LSTM with dropout and enhanced initialization
            self.temporal_lstm = nn.LSTM(
                input_size=self.standard_dim * self.num_features,
                hidden_size=self.standard_dim,
                batch_first=True,
                bidirectional=True,
                num_layers=1
            )
            
            # Residual connection projection layer
            self.residual_proj = nn.Linear(self.standard_dim * self.num_features, self.standard_dim * 2)
            
            # Final fusion layer normalization
            self.final_norm = nn.LayerNorm(self.standard_dim * 2)
            
            # Reapply initialization
            self._init_weights()
            
        def _init_weights(self):
            """Improved weight initialization strategy"""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    # Apply Xavier uniform initialization
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LSTM):
                    # Specialized LSTM weight initialization
                    for name, param in module.named_parameters():
                        if 'weight_ih' in name:
                            nn.init.xavier_uniform_(param.data)
                        elif 'weight_hh' in name:
                            nn.init.orthogonal_(param.data)
                        elif 'bias' in name:
                            nn.init.zeros_(param.data)
                            # Set forget gate bias to one
                            n = param.size(0)
                            param.data[n//4:n//2].fill_(1.)
            
        def forward(self, features):
            """
            Forward propagation through fusion layers
            """
            batch_size, seq_len, _ = features[self.feature_types[0]].shape
            
            # 1. Feature normalization and transformation
            transformed_features = {}
            for feat_type in self.feature_types:
                normalized = self.feature_norms[feat_type](features[feat_type])
                transformed = self.feature_transforms[feat_type](normalized)
                transformed_features[feat_type] = transformed
            
            # 2. Multi grained gated fusion
            multi_grained_features = []
            gate_weights = {feat_type: [] for feat_type in self.feature_types}

            # Enforce strict positivity via exponential activation preventing saturation gradients
            current_temp = torch.exp(self.log_temp)
            
            for i in range(self.num_groups):
                # Extract features for current group slice
                start_idx = i * self.group_size
                end_idx = (i + 1) * self.group_size
                
                group_feats = []
                for feat_type in self.feature_types:
                    group_feats.append(transformed_features[feat_type][..., start_idx:end_idx])
                
                # Concatenate features and compute gating weights
                group_concat = torch.cat(group_feats, dim=-1)
                group_logits = self.group_gate_nets[i](group_concat)
                
                # Apply temperature scaled softmax to improve numerical stability
                group_gates = F.softmax(group_logits / current_temp, dim=-1)
                
                # Multiply features by corresponding gate weights
                weighted_feats = []
                for j, feat_type in enumerate(self.feature_types):
                    weight = group_gates[..., j:j+1]
                    gate_weights[feat_type].append(weight)
                    weighted_feat = group_feats[j] * weight
                    weighted_feats.append(weighted_feat)
                
                # Concatenate weighted features
                group_feature = torch.cat(weighted_feats, dim=-1)
                multi_grained_features.append(group_feature)
            
            # Concatenate features across all groups
            multi_grained_fusion = torch.cat(multi_grained_features, dim=-1)
            
            # Compute average weight per feature for monitoring purposes
            avg_weights = {}
            for feat_type in self.feature_types:
                avg_weights[feat_type] = torch.cat(gate_weights[feat_type], dim=-1).mean(dim=-1, keepdim=True)
            
            # 3. Temporal processing integrating residual connections
            residual_input = self.residual_proj(multi_grained_fusion)
            
            temporal_features, _ = self.temporal_lstm(multi_grained_fusion)
            
            # Add residual representation
            temporal_features = temporal_features + residual_input
            
            # 4. Final feature projection
            fused_features = self.final_norm(temporal_features)
            
            return fused_features, avg_weights, current_temp
            
        def get_fusion_weights(self):
            """
            Retrieve currently utilized gating fusion strategy weights
            """
            return {
                "grain_weight": 0.5,
                "temporal_weight": 0.5
            } weights
            """
            return {
                "grain_weight": 0.5,
                "temporal_weight": 0.5
            }