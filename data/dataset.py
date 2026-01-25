#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset classes for spectral data.
Provides PyTorch Dataset implementation for multispectral imagery.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple


class SpectralDataset(Dataset):
    """
    PyTorch Dataset for multispectral data.
    
    Args:
        X_spectral: Spectral data tensor [N, C, H, W]
        y: Labels tensor [N,] or [N, num_classes]
        is_training: Whether this is a training dataset
    """
    
    def __init__(self, X_spectral: torch.Tensor, y: Optional[torch.Tensor] = None, 
                 is_training: bool = True):
        super().__init__()
        
        self.X_spectral = X_spectral.float()
        self.y = self._process_labels(y) if y is not None else None
        self.is_training = is_training
        
    def _process_labels(self, y: torch.Tensor) -> torch.Tensor:
        """
        Process labels to ensure correct format.
        
        Args:
            y: Input labels
            
        Returns:
            Processed labels as long tensor
        """
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        
        # Convert one-hot to class indices if needed
        if y.dim() > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)
        
        return y.long()
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.X_spectral)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (spectral_data, label)
        """
        spectral = self.X_spectral[idx].clone()
        
        if self.y is not None:
            label = self.y[idx].clone()
            return spectral, label
        else:
            return spectral, torch.tensor(-1)
    
    @property
    def num_bands(self) -> int:
        """Return the number of spectral bands."""
        return self.X_spectral.shape[1]
    
    @property
    def spatial_size(self) -> Tuple[int, int]:
        """Return the spatial dimensions (H, W)."""
        return self.X_spectral.shape[2], self.X_spectral.shape[3]
    
    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        if self.y is None:
            return 0
        return len(torch.unique(self.y))