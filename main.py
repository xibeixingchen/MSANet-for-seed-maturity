#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
纯光谱深度学习框架 - 主程序
专注于光谱数据分析，包含注意力机制和完整的评估体系
"""

import os
import argparse
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import warnings

from data import SpectralDataset, load_spectral_data
from training import train_spectral_model
from utils import setup_logging, set_seed, save_results

warnings.filterwarnings('ignore')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='纯光谱深度学习框架')
    
    # 数据参数
    parser.add_argument('--data-path', required=True, help='光谱数据文件路径(.npz)')
    parser.add_argument('--num-bands', type=int, default=19, help='光谱波段数')
    parser.add_argument('--num-classes', type=int, default=5, help='分类类别数')
    
    # 模型参数
    parser.add_argument('--feature-dim', type=int, default=256, help='特征维度')
    parser.add_argument('--dropout-rate', type=float, default=0.15, help='Dropout率')
    parser.add_argument('--spectral-attention-reduction', type=int, default=8, help='光谱注意力降维倍数')
    parser.add_argument('--spatial-attention-heads', type=int, default=8, help='空间注意力头数')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    
    # 其他参数
    parser.add_argument('--output-dir', default='./spectral_output', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--test-split', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--val-split', type=float, default=0.2, help='验证集比例')
    
    args = parser.parse_args()
    
    # 设置种子
    set_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"spectral_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    setup_logging(os.path.join(args.output_dir, 'spectral_training.log'))
    
    import logging
    logging.info("纯光谱深度学习框架")
    logging.info(f"配置: {vars(args)}")
    
    try:
        # 加载数据
        logging.info("加载光谱数据...")
        X_spectral, y = load_spectral_data(args.data_path)
        
        if X_spectral is None:
            logging.error("数据加载失败")
            return
        
        # 数据集分割
        # 首先分出测试集
        indices = np.arange(len(X_spectral))
        if y.dim() > 1 and y.shape[1] > 1:
            y_indices = y.argmax(dim=1).numpy()
        else:
            y_indices = y.numpy()
        
        train_val_indices, test_indices = train_test_split(
            indices, test_size=args.test_split, stratify=y_indices, random_state=42
        )
        
        # 再从训练集中分出验证集
        train_indices, val_indices = train_test_split(
            train_val_indices, 
            test_size=args.val_split/(1-args.test_split), 
            stratify=y_indices[train_val_indices], 
            random_state=42
        )
        
        # 分割数据
        X_train, X_val, X_test = X_spectral[train_indices], X_spectral[val_indices], X_spectral[test_indices]
        y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
        
        logging.info(f"数据集分割: 训练={len(train_indices)}, 验证={len(val_indices)}, 测试={len(test_indices)}")
        
        # 创建数据集和数据加载器
        train_dataset = SpectralDataset(X_train, y_train, is_training=True)
        val_dataset = SpectralDataset(X_val, y_val, is_training=False)
        test_dataset = SpectralDataset(X_test, y_test, is_training=False)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # 训练模型
        model, history, best_val_acc, test_metrics, test_targets, test_preds = train_spectral_model(
            args, train_loader, val_loader, test_loader
        )
        
        # 保存结果
        save_results(args, history, best_val_acc, test_metrics, test_targets, test_preds)
        
    except Exception as e:
        import logging
        import traceback
        logging.error(f"训练过程出错: {str(e)}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()