#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练模块
包含训练器和配置管理
"""

from .trainer import train_spectral_model
from .config import create_default_config

__all__ = [
    'train_spectral_model',
    'create_default_config'
]