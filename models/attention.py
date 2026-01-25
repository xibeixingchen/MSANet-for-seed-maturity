#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Attention Mechanism Module
Contains spectral attention and spatial attention mechanisms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralAttentionBlock(nn.Module):
    """Spectral attention block - core component"""
    
    def __init__(self, num_bands, reduction=8):
        super(SpectralAttentionBlock, self).__init__()
        
        self.num_bands = num_bands
        hidden_dim = max(num_bands // reduction, 4)
        
        # Global pooling branch - captures global spectral features
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_bands, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_bands, 1, bias=False)
        )
        
        # Local convolution branch - captures local spatial-spectral features
        self.local_branch = nn.Sequential(
            nn.Conv2d(num_bands, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_bands, 1, bias=False)
        )
        
        # Adaptive fusion weight
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # Global attention
        global_attn = self.global_branch(x)
        
        # Local attention  
        local_attn = self.local_branch(x)
        
        # Adaptive fusion
        alpha = torch.sigmoid(self.fusion_weight)
        combined_attn = alpha * global_attn + (1 - alpha) * local_attn
        
        # Apply attention weights
        attention_weights = torch.sigmoid(combined_attn)
        attended_x = x * attention_weights
        
        return attended_x, attention_weights.mean(dim=(2, 3))


class SpatialAttentionProcessor(nn.Module):
    """Spatial attention processor"""
    
    def __init__(self, dim, num_heads=8):
        super(SpatialAttentionProcessor, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # Position encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, 16, 16))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Add position encoding
        pos = F.interpolate(self.pos_embed, size=(h, w), mode='bilinear', align_corners=False)
        x = x + pos
        
        # Reshape to sequence format: [B, C, H, W] -> [B, H*W, C]
        x_seq = x.flatten(2).transpose(1, 2)
        
        # Self-attention
        x_norm = self.norm1(x_seq)
        attn_out, attn_weights = self.attention(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out
        
        # Feed-forward network
        x_seq = x_seq + self.ffn(self.norm2(x_seq))
        
        # Reshape back to feature map: [B, H*W, C] -> [B, C, H, W]
        x_out = x_seq.transpose(1, 2).reshape(b, c, h, w)
        
        return x_out, attn_weights