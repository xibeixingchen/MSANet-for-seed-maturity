#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data module for spectral analysis framework.
Provides dataset classes and data loading utilities.
"""

from .dataset import SpectralDataset
from .preprocessing import load_spectral_data

__all__ = ['SpectralDataset', 'load_spectral_data']