#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backbone Network Module
Contains 3D CNN and related base network components
"""
import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    """3D convolution block - processes spectral-spatial information"""
    
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1)):
        super(Conv3DBlock, self).__init__()
        
        padding = tuple(k // 2 for k in kernel_size)
        
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=False
        )
        self.bn3d = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv3d(x)
        x = self.bn3d(x)
        x = self.relu(x)
        return x


class SpectralCNN(nn.Module):
    """3D CNN backbone - designed for spectral data"""
    
    def __init__(self, num_bands, feature_dim=256):
        super(SpectralCNN, self).__init__()
        
        self.num_bands = num_bands
        self.feature_dim = feature_dim
        
        # 3D convolution layers: process spectral-spatial information
        self.conv3d_layers = nn.ModuleList([
            Conv3DBlock(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1)),      # Preserve spectral dimension
            Conv3DBlock(32, 64, kernel_size=(3, 3, 3), stride=(2, 1, 1)),     # Downsample spectral dimension
            Conv3DBlock(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2)),    # Downsample spatial dimension
        ])
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 7, 7))
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Conv2d(128, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Add channel dimension for 3D convolution: [B, C, H, W] -> [B, 1, C, H, W]
        x = x.unsqueeze(1)
        
        # 3D convolution feature extraction
        for conv_layer in self.conv3d_layers:
            x = conv_layer(x)
        
        # Adaptive pooling: [B, 128, C', H', W'] -> [B, 128, 1, 7, 7]
        x = self.adaptive_pool(x)
        
        # Remove spectral dimension: [B, 128, 1, 7, 7] -> [B, 128, 7, 7]
        x = x.squeeze(2)
        
        # Feature projection
        x = self.feature_projection(x)
        
        return x