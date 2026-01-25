#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading and preprocessing utilities for spectral data.
Provides functions to load NPZ format multispectral datasets.
"""

import os
import re
import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, List
import cv2


def load_spectral_data(file_path: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Load spectral data from NPZ or PT file.
    
    Args:
        file_path: Path to data file (.npz or .pt)
        
    Returns:
        Tuple of (spectral_tensor, labels_tensor) or (None, None) if loading fails
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return None, None
        
        logging.info(f"Loading data from: {file_path}")
        
        if file_path.endswith('.pt'):
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            
            spectral_tensor = data.get('spectral', data.get('X'))
            labels_tensor = data.get('labels', data.get('y'))
            
            if spectral_tensor is None or labels_tensor is None:
                logging.error(f"Required data not found. Keys: {list(data.keys())}")
                return None, None
            
            if not isinstance(spectral_tensor, torch.Tensor):
                spectral_tensor = torch.from_numpy(spectral_tensor)
            if not isinstance(labels_tensor, torch.Tensor):
                labels_tensor = torch.from_numpy(labels_tensor)
            
            spectral_tensor = spectral_tensor.float()
            
        else:
            data = np.load(file_path, allow_pickle=True)
            
            spectral_data = None
            labels_data = None
            spectral_keys = ['X', 'spectral', 'hyper', 'hyperspectral', 'multispectral']
            label_keys = ['y', 'labels', 'targets', 'label', 'target']
            
            for key in spectral_keys:
                if key in data.keys():
                    spectral_data = data[key]
                    break
            
            if spectral_data is None:
                for key in data.keys():
                    if 'spectral' in key.lower():
                        spectral_data = data[key]
                        break
            
            for key in label_keys:
                if key in data.keys():
                    labels_data = data[key]
                    break
            
            if labels_data is None:
                for key in data.keys():
                    if 'label' in key.lower():
                        labels_data = data[key]
                        break
            
            if spectral_data is None or labels_data is None:
                logging.error(f"Required data not found. Keys: {list(data.keys())}")
                return None, None
            
            spectral_tensor = torch.from_numpy(spectral_data).float()
            labels_tensor = torch.from_numpy(labels_data)
        
        if spectral_tensor.dim() == 4:
            n, d1, d2, d3 = spectral_tensor.shape
            if d3 < d1 and d3 < d2:
                logging.info(f"Transposing data from [N, H, W, C] to [N, C, H, W]")
                spectral_tensor = spectral_tensor.permute(0, 3, 1, 2)
        
        if labels_tensor.dim() > 1 and labels_tensor.shape[1] > 1:
            labels_tensor = torch.argmax(labels_tensor, dim=1)
        
        labels_tensor = labels_tensor.long()
        
        logging.info(f"Spectral shape: {spectral_tensor.shape}")
        logging.info(f"Labels shape: {labels_tensor.shape}")
        logging.info(f"Data range: [{spectral_tensor.min():.4f}, {spectral_tensor.max():.4f}]")
        logging.info(f"Classes: {len(torch.unique(labels_tensor))}, Distribution: {torch.bincount(labels_tensor).tolist()}")
        
        return spectral_tensor, labels_tensor
        
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None


def normalize_spectral_data(spectral_data: torch.Tensor, 
                            method: str = 'minmax') -> torch.Tensor:
    """
    Normalize spectral data.
    
    Args:
        spectral_data: Input tensor [N, C, H, W]
        method: Normalization method ('minmax', 'zscore', 'percentile')
        
    Returns:
        Normalized tensor
    """
    if method == 'minmax':
        # Per-sample min-max normalization
        data_min = spectral_data.view(spectral_data.size(0), -1).min(dim=1, keepdim=True)[0]
        data_max = spectral_data.view(spectral_data.size(0), -1).max(dim=1, keepdim=True)[0]
        data_min = data_min.view(-1, 1, 1, 1)
        data_max = data_max.view(-1, 1, 1, 1)
        
        normalized = (spectral_data - data_min) / (data_max - data_min + 1e-8)
        
    elif method == 'zscore':
        # Per-sample z-score normalization
        mean = spectral_data.view(spectral_data.size(0), -1).mean(dim=1, keepdim=True)
        std = spectral_data.view(spectral_data.size(0), -1).std(dim=1, keepdim=True)
        mean = mean.view(-1, 1, 1, 1)
        std = std.view(-1, 1, 1, 1)
        
        normalized = (spectral_data - mean) / (std + 1e-8)
        
    elif method == 'percentile':
        # Per-sample percentile normalization (robust to outliers)
        flat_data = spectral_data.view(spectral_data.size(0), -1)
        p_low = torch.quantile(flat_data, 0.01, dim=1, keepdim=True)
        p_high = torch.quantile(flat_data, 0.99, dim=1, keepdim=True)
        p_low = p_low.view(-1, 1, 1, 1)
        p_high = p_high.view(-1, 1, 1, 1)
        
        normalized = torch.clamp((spectral_data - p_low) / (p_high - p_low + 1e-8), 0, 1)
        
    else:
        logging.warning(f"Unknown normalization method: {method}, returning original data")
        normalized = spectral_data
    
    return normalized


def get_band_statistics(spectral_data: torch.Tensor) -> dict:
    """
    Compute statistics for each spectral band.
    
    Args:
        spectral_data: Input tensor [N, C, H, W]
        
    Returns:
        Dictionary containing per-band statistics
    """
    num_bands = spectral_data.shape[1]
    stats = {
        'mean': [],
        'std': [],
        'min': [],
        'max': []
    }
    
    for b in range(num_bands):
        band_data = spectral_data[:, b, :, :]
        stats['mean'].append(band_data.mean().item())
        stats['std'].append(band_data.std().item())
        stats['min'].append(band_data.min().item())
        stats['max'].append(band_data.max().item())
    
    return stats


# =============================================================================
# Raw data preprocessing functions
# =============================================================================

def _extract_stage(file_path: str) -> Optional[str]:
    """Extract stage (S1-S6 or D1-D7) from file path."""
    file_path = str(file_path).replace('\\', '/')
    
    stage_match = re.search(r'/(S[1-6]|D[1-7])(?:_(?:up|down))?(?:/|$)', file_path, re.IGNORECASE)
    if stage_match:
        return stage_match.group(1).upper()
    
    filename = os.path.basename(file_path)
    stage_match = re.search(r'[_-](S[1-6]|D[1-7])([_-]|$)', filename, re.IGNORECASE)
    if stage_match:
        return stage_match.group(1).upper()
    
    loose_match = re.search(r'(S[1-6]|D[1-7])', file_path, re.IGNORECASE)
    if loose_match:
        return loose_match.group(1).upper()
    
    return None


def _discover_stages(directory: str) -> List[str]:
    """Scan directory to discover available stages."""
    stages_found = set()
    
    for path in Path(directory).rglob('*'):
        path_str = str(path).replace('\\', '/')
        matches = re.findall(r'[/\._-](S[1-6]|D[1-7])(?:_(?:up|down))?(?:[/\._-]|$)', path_str, re.IGNORECASE)
        for match in matches:
            clean_stage = match.upper()
            if clean_stage[0] in ['S', 'D']:
                stages_found.add(clean_stage)
    
    def sort_key(stage):
        return (0 if stage[0] == 'S' else 1, int(stage[1:]))
    
    return sorted(list(stages_found), key=sort_key)


def _preprocess_spectral_array(spectral_data: np.ndarray, 
                                target_size: Tuple[int, int] = (224, 224),
                                num_bands: int = 19) -> Optional[np.ndarray]:
    """Preprocess spectral array to [C, H, W] format."""
    try:
        if spectral_data.ndim < 2:
            return None
        
        if spectral_data.ndim == 2:
            spectral_data = spectral_data[:, :, np.newaxis]
        
        current_bands = spectral_data.shape[2]
        
        if current_bands != num_bands:
            if current_bands > num_bands:
                indices = np.linspace(0, current_bands - 1, num_bands, dtype=int)
                spectral_data = spectral_data[:, :, indices]
            else:
                padding = np.repeat(spectral_data[:, :, -1:], num_bands - current_bands, axis=2)
                spectral_data = np.concatenate([spectral_data, padding], axis=2)
        
        result = np.zeros((num_bands, *target_size), dtype=np.float32)
        
        for i in range(num_bands):
            band = spectral_data[:, :, i].astype(np.float32)
            band = np.nan_to_num(band, nan=0.0, posinf=1.0, neginf=0.0)
            resized = cv2.resize(band, target_size, interpolation=cv2.INTER_LINEAR)
            
            p_low, p_high = np.percentile(resized, [1, 99])
            if p_high <= p_low:
                p_high = p_low + 1e-6
            result[i] = np.clip((resized - p_low) / (p_high - p_low), 0, 1)
        
        return result
    except:
        return None


def preprocess_raw_directory(data_dir: str, output_path: str,
                             image_size: int = 224, num_bands: int = 19) -> bool:
    """
    Preprocess raw spectral data from directory and save as NPZ.
    
    Args:
        data_dir: Directory containing raw .npy spectral files
        output_path: Output NPZ file path
        image_size: Target image size
        num_bands: Number of spectral bands
        
    Returns:
        True if successful
    """
    logging.info(f"Processing raw data from: {data_dir}")
    
    stages = _discover_stages(data_dir)
    if not stages:
        logging.error("No stages found in directory")
        return False
    
    logging.info(f"Found stages: {stages}")
    stage_to_idx = {stage: idx for idx, stage in enumerate(stages)}
    
    all_files = list(Path(data_dir).rglob('*_spectral_roi.npy'))
    logging.info(f"Found {len(all_files)} spectral ROI files")
    
    all_spectral = []
    all_labels = []
    stage_counts = {s: 0 for s in stages}
    
    for f in all_files:
        stage = _extract_stage(str(f))
        if stage not in stage_to_idx:
            continue
        
        try:
            raw_data = np.load(str(f), allow_pickle=True)
            processed = _preprocess_spectral_array(raw_data, (image_size, image_size), num_bands)
            
            if processed is not None:
                all_spectral.append(processed)
                all_labels.append(stage_to_idx[stage])
                stage_counts[stage] += 1
        except Exception as e:
            logging.warning(f"Failed to process {f}: {e}")
            continue
    
    if not all_spectral:
        logging.error("No valid data processed")
        return False
    
    X = np.array(all_spectral, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    logging.info(f"Final data shape: X={X.shape}, y={y.shape}")
    logging.info(f"Stage counts: {stage_counts}")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y, spectral=X, labels=y)
    logging.info(f"Saved to: {output_path}")
    
    meta_path = output_path.replace('.npz', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'num_samples': len(X),
            'num_classes': len(stages),
            'stages': stages,
            'stage_to_idx': stage_to_idx,
            'stage_counts': stage_counts,
            'shape': list(X.shape)
        }, f, indent=2)
    
    return True


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    parser = argparse.ArgumentParser(description='Preprocess raw spectral data to NPZ')
    parser.add_argument('--data-dir', required=True, help='Raw data directory')
    parser.add_argument('--output', required=True, help='Output NPZ path')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--num-bands', type=int, default=19)
    
    args = parser.parse_args()
    
    success = preprocess_raw_directory(args.data_dir, args.output, args.image_size, args.num_bands)
    exit(0 if success else 1)