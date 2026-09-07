"""
Model definition module
Defines VAD model and its configuration supporting multi-grained and temporal gating mechanisms
"""

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from model_components import ModelComponents

class VADConfig(PretrainedConfig):
    """
    VAD model configuration class
    """
    def __init__(
        self,
        emotion2vec_dim=1024,
        hubert_dim=1024,
        hidden_dim=1024,
        intermediate_dim=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        hidden_dropout_prob=0.1,
        use_multi_grained_gating=True,
        use_temporal_gating=True,
        num_groups=8,
        gating_temperature=0.1,
        wav2vec_dim=0,         # wav2vec feature dimension where 0 indicates disabled
        data2vec_dim=0,       # data2vec feature dimension where 0 indicates disabled
        num_emotions=8,  # Number of discrete emotion categories defaulting to 8
        **kwargs
    ):
        """
        Initialize VAD configuration parameters
        """
        super().__init__(**kwargs)
        self.emotion2vec_dim = emotion2vec_dim
        self.hubert_dim = hubert_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_multi_grained_gating = use_multi_grained_gating
        self.use_temporal_gating = use_temporal_gating
        self.num_groups = num_groups
        self.gating_temperature = gating_temperature
        self.wav2vec_dim = wav2vec_dim
        self.data2vec_dim = data2vec_dim
        self.num_emotions = num_emotions

class VADModelWithGating(PreTrainedModel):
    """
    VAD model with gating mechanism
    Supports gating mechanism across multiple feature inputs
    """
    def __init__(self, config):
        """
        Initialize VAD model
        """
        super().__init__(config)
        self.config = config
        
        # Determine feature types and their dimensions
        feature_dims = {'emotion2vec': config.emotion2vec_dim, 'hubert': config.hubert_dim}
        
        # Include supplementary features if specified in configuration
        if hasattr(config, 'wav2vec_dim') and config.wav2vec_dim > 0:
            feature_dims['wav2vec'] = config.wav2vec_dim
        if hasattr(config, 'data2vec_dim') and config.data2vec_dim > 0:
            feature_dims['data2vec'] = config.data2vec_dim
            
        self.feature_types = list(feature_dims.keys())
        self.num_features = len(self.feature_types)
        
        # Apply enhanced gated feature fusion
        self.feature_fusion = ModelComponents.GatedFeatureFusion(
            feature_dims=feature_dims,
            num_groups=config.num_groups,
            temperature=config.gating_temperature
        )
        
        # Weighted-sum fusion has a fixed output size regardless of how many
        # feature sources are active. The bidirectional LSTM doubles it.
        fusion_output_dim = self.feature_fusion.output_dim
        
        # Setup input projection layer dimension
        self.input_proj = nn.Linear(fusion_output_dim, config.hidden_dim)
        
        # Initialize Transformer encoder layers with Pre-Norm standard
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_dim,
            dropout=config.hidden_dropout_prob,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        
        # Setup output layers and pooling
        self.pooler = ModelComponents.AttentionPooling(config.hidden_dim)

        self.output_proj_vad = nn.Linear(config.hidden_dim, 3)
        
        # Add nonlinear projection head for contrastive learning inspired by SimCLR architecture
        self.contrastive_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 128)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, features, padding_mask=None):
        """
        Forward pass computing VAD output gate weights and contrastive features
        """
        # Execute feature fusion
        x, gate_weights, current_temp = self.feature_fusion(features, padding_mask)
        
        # Project fused features to hidden dimension
        x = self.input_proj(x)
        x = self.dropout(x)
        
        # Process through Transformer encoder
        x = self.encoder_layers(x, src_key_padding_mask=padding_mask)
            
        # Extract pooled features and compute VAD predictions bounded between 0 and 1
        pooled_features = self.pooler(x, padding_mask)
        vad_output = torch.sigmoid(self.output_proj_vad(pooled_features))
        
        # Extract features designated for contrastive learning
        contrastive_features = self.contrastive_proj(pooled_features)
        
        return vad_output, gate_weights, contrastive_features, current_temp
        
    def get_fusion_weights(self):
        """
        Retrieve weights utilized by the current gated fusion strategy
        """
        if hasattr(self.feature_fusion, 'get_fusion_weights'):
            return self.feature_fusion.get_fusion_weights()
        return None
