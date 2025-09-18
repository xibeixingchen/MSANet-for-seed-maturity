#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
注意力机制模块
包含光谱注意力和空间注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralAttentionBlock(nn.Module):
    """光谱注意力块 - 核心组件"""
    
    def __init__(self, num_bands, reduction=8):
        super(SpectralAttentionBlock, self).__init__()
        
        self.num_bands = num_bands
        hidden_dim = max(num_bands // reduction, 4)
        
        # 全局池化分支 - 捕获全局光谱特征
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_bands, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_bands, 1, bias=False)
        )
        
        # 局部卷积分支 - 捕获局部空间-光谱特征
        self.local_branch = nn.Sequential(
            nn.Conv2d(num_bands, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_bands, 1, bias=False)
        )
        
        # 自适应融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # 全局注意力
        global_attn = self.global_branch(x)
        
        # 局部注意力  
        local_attn = self.local_branch(x)
        
        # 自适应融合
        alpha = torch.sigmoid(self.fusion_weight)
        combined_attn = alpha * global_attn + (1 - alpha) * local_attn
        
        # 应用注意力权重
        attention_weights = torch.sigmoid(combined_attn)
        attended_x = x * attention_weights
        
        return attended_x, attention_weights.mean(dim=(2, 3))


class SpatialAttentionProcessor(nn.Module):
    """空间注意力处理器"""
    
    def __init__(self, dim, num_heads=8):
        super(SpatialAttentionProcessor, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, 16, 16))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 多头自注意力
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 添加位置编码
        pos = F.interpolate(self.pos_embed, size=(h, w), mode='bilinear', align_corners=False)
        x = x + pos
        
        # 重塑为序列格式: [B, C, H, W] -> [B, H*W, C]
        x_seq = x.flatten(2).transpose(1, 2)
        
        # 自注意力
        x_norm = self.norm1(x_seq)
        attn_out, attn_weights = self.attention(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out
        
        # 前馈网络
        x_seq = x_seq + self.ffn(self.norm2(x_seq))
        
        # 重塑回特征图: [B, H*W, C] -> [B, C, H, W]
        x_out = x_seq.transpose(1, 2).reshape(b, c, h, w)
        
        return x_out, attn_weights