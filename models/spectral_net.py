#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spectral Network Main Model
Contains complete SpectralNet architecture and classification head
"""

import torch
import torch.nn as nn
from .attention import SpectralAttentionBlock, SpatialAttentionProcessor
from .backbone import SpectralCNN


class SpectralClassificationHead(nn.Module):
    """Spectral classification head"""
    
    def __init__(self, input_dim, num_classes, dropout_rate=0.15):
        super(SpectralClassificationHead, self).__init__()
        
        # Multi-scale global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.gmp = nn.AdaptiveMaxPool2d(1)  # Global max pooling
        
        # Classifier
        pool_dim = input_dim * 2  # GAP + GMP
        
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(input_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # Multi-scale pooling
        gap_feat = self.gap(x).flatten(1)
        gmp_feat = self.gmp(x).flatten(1)
        
        # Feature concatenation
        combined = torch.cat([gap_feat, gmp_feat], dim=1)
        
        # Classification
        output = self.classifier(combined)
        
        return output


class SpectralNet(nn.Module):
    """Complete spectral analysis network"""
    
    def __init__(self, num_bands=19, num_classes=5, config=None):
        super(SpectralNet, self).__init__()
        
        if config is None:
            config = {
                'feature_dim': 256,
                'dropout_rate': 0.15,
                'spectral_attention_reduction': 8,
                'spatial_attention_heads': 8
            }
        
        self.config = config
        self.num_bands = num_bands
        self.num_classes = num_classes
        
        # Input normalization
        self.input_norm = nn.BatchNorm2d(num_bands)
        
        # Spectral attention module
        self.spectral_attention = SpectralAttentionBlock(
            num_bands, 
            reduction=config['spectral_attention_reduction']
        )
        
        # 3D CNN backbone
        self.backbone_3d = SpectralCNN(
            num_bands=num_bands,
            feature_dim=config['feature_dim']
        )
        
        # Spatial attention processor
        self.spatial_processor = SpatialAttentionProcessor(
            dim=config['feature_dim'], 
            num_heads=config['spatial_attention_heads']
        )
        
        # Classification head
        self.classifier = SpectralClassificationHead(
            input_dim=config['feature_dim'],
            num_classes=num_classes,
            dropout_rate=config['dropout_rate']
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x, return_features=False):
        """Forward pass"""
        features = {}
        
        # Input normalization
        x = self.input_norm(x)
        features['input_normalized'] = x
        
        # Spectral attention
        x_attended, spectral_weights = self.spectral_attention(x)
        features['spectral_attended'] = x_attended
        features['spectral_weights'] = spectral_weights
        
        # 3D CNN feature extraction
        cnn_features = self.backbone_3d(x_attended)
        features['cnn_features'] = cnn_features
        
        # Spatial attention processing
        spatial_features, spatial_attention = self.spatial_processor(cnn_features)
        features['spatial_features'] = spatial_features
        features['spatial_attention'] = spatial_attention
        
        # Classification
        output = self.classifier(spatial_features)
        features['classification_output'] = output
        
        if return_features:
            return output, features
        return output