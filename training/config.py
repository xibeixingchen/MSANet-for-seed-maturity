#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置管理模块
定义模型和训练的默认配置
"""


def create_default_config():
    """创建默认配置"""
    config = {
        # 模型参数
        'feature_dim': 256,
        'dropout_rate': 0.15,
        'spectral_attention_reduction': 8,
        'spatial_attention_heads': 8,
        
        # 训练参数
        'batch_size': 16,
        'epochs': 100,
        'lr': 0.001,
        'weight_decay': 0.01,
        'patience': 15,
        
        # 数据参数
        'test_split': 0.1,
        'val_split': 0.2,
        
        # 其他参数
        'seed': 42,
    }
    return config


def get_optimizer_config():
    """获取优化器配置"""
    return {
        'type': 'AdamW',
        'lr': 0.001,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999)
    }


def get_scheduler_config():
    """获取学习率调度器配置"""
    return {
        'type': 'CosineAnnealingWarmRestarts',
        'T_0': 50,  # epochs // 2
        'T_mult': 2,
        'eta_min_ratio': 0.01
    }