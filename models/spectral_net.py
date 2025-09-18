#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
光谱网络主模型
包含完整的SpectralNet架构和分类头
"""

import torch
import torch.nn as nn
from .attention import SpectralAttentionBlock, SpatialAttentionProcessor
from .backbone import SpectralCNN


class SpectralClassificationHead(nn.Module):
    """光谱分类头"""
    
    def __init__(self, input_dim, num_classes, dropout_rate=0.15):
        super(SpectralClassificationHead, self).__init__()
        
        # 多尺度全局池化
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.gmp = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
        
        # 分类器
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
        # 多尺度池化
        gap_feat = self.gap(x).flatten(1)
        gmp_feat = self.gmp(x).flatten(1)
        
        # 特征拼接
        combined = torch.cat([gap_feat, gmp_feat], dim=1)
        
        # 分类
        output = self.classifier(combined)
        
        return output


class SpectralNet(nn.Module):
    """完整的光谱分析网络"""
    
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
        
        # 输入标准化
        self.input_norm = nn.BatchNorm2d(num_bands)
        
        # 光谱注意力模块
        self.spectral_attention = SpectralAttentionBlock(
            num_bands, 
            reduction=config['spectral_attention_reduction']
        )
        
        # 3D CNN骨干网络
        self.backbone_3d = SpectralCNN(
            num_bands=num_bands,
            feature_dim=config['feature_dim']
        )
        
        # 空间注意力处理器
        self.spatial_processor = SpatialAttentionProcessor(
            dim=config['feature_dim'], 
            num_heads=config['spatial_attention_heads']
        )
        
        # 分类头
        self.classifier = SpectralClassificationHead(
            input_dim=config['feature_dim'],
            num_classes=num_classes,
            dropout_rate=config['dropout_rate']
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
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
        """前向传播"""
        features = {}
        
        # 输入标准化
        x = self.input_norm(x)
        features['input_normalized'] = x
        
        # 光谱注意力
        x_attended, spectral_weights = self.spectral_attention(x)
        features['spectral_attended'] = x_attended
        features['spectral_weights'] = spectral_weights
        
        # 3D CNN特征提取
        cnn_features = self.backbone_3d(x_attended)
        features['cnn_features'] = cnn_features
        
        # 空间注意力处理
        spatial_features, spatial_attention = self.spatial_processor(cnn_features)
        features['spatial_features'] = spatial_features
        features['spatial_attention'] = spatial_attention
        
        # 分类
        output = self.classifier(spatial_features)
        features['classification_output'] = output
        
        if return_features:
            return output, features
        return output