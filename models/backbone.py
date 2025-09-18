#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
骨干网络模块
包含3D CNN和相关的基础网络组件
"""

import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    """3D卷积块 - 处理光谱-空间信息"""
    
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
    """3D CNN骨干网络 - 专用于光谱数据"""
    
    def __init__(self, num_bands, feature_dim=256):
        super(SpectralCNN, self).__init__()
        
        self.num_bands = num_bands
        self.feature_dim = feature_dim
        
        # 3D卷积层序列：处理光谱-空间信息
        self.conv3d_layers = nn.ModuleList([
            Conv3DBlock(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1)),      # 保持光谱维度
            Conv3DBlock(32, 64, kernel_size=(3, 3, 3), stride=(2, 1, 1)),     # 降采样光谱维度
            Conv3DBlock(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2)),    # 降采样空间维度
        ])
        
        # 自适应池化到固定尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 7, 7))
        
        # 特征投影
        self.feature_projection = nn.Sequential(
            nn.Conv2d(128, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 添加通道维度用于3D卷积: [B, C, H, W] -> [B, 1, C, H, W]
        x = x.unsqueeze(1)
        
        # 3D卷积特征提取
        for conv_layer in self.conv3d_layers:
            x = conv_layer(x)
        
        # 自适应池化: [B, 128, C', H', W'] -> [B, 128, 1, 7, 7]
        x = self.adaptive_pool(x)
        
        # 移除光谱维度: [B, 128, 1, 7, 7] -> [B, 128, 7, 7]
        x = x.squeeze(2)
        
        # 特征投影
        x = self.feature_projection(x)
        
        return x