#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化模块
包含结果保存和可视化功能
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging


def save_results(args, history, best_val_acc, test_metrics, all_targets, all_preds):
    """保存训练结果"""
    
    # 保存结果JSON
    results = {
        'best_val_accuracy': best_val_acc,
        'test_metrics': test_metrics,
        'config': {
            'num_bands': args.num_bands,
            'num_classes': args.num_classes,
            'feature_dim': args.feature_dim,
            'dropout_rate': args.dropout_rate,
            'lr': args.lr,
            'epochs': args.epochs
        },
        'training_history': history
    }
    
    with open(os.path.join(args.output_dir, 'spectral_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存训练历史CSV
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'train_f1': history['train_f1'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc'],
        'val_f1': history['val_f1'],
        'learning_rate': history['learning_rate']
    })
    history_df.to_csv(os.path.join(args.output_dir, 'training_history.csv'), index=False)
    
    # 保存预测结果CSV
    predictions_df = pd.DataFrame({
        'sample_id': range(len(all_targets)),
        'true_label': all_targets,
        'predicted_label': all_preds,
        'correct': np.array(all_targets) == np.array(all_preds)
    })
    predictions_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
    
    # 绘制训练历史
    plot_training_curves(history, args.output_dir)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(all_targets, all_preds, args.output_dir)
    
    logging.info(f"结果保存完成: {args.output_dir}")


def plot_training_curves(history, output_dir):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train', linewidth=2)
    plt.plot(history['val_loss'], label='Validation', linewidth=2)
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train', linewidth=2)
    plt.plot(history['val_acc'], label='Validation', linewidth=2)
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='Train', linewidth=2)
    plt.plot(history['val_f1'], label='Validation', linewidth=2)
    plt.title('F1-Score History')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.pdf'), bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(all_targets, all_preds, output_dir):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(cm.shape[1])],
                yticklabels=[f'Class {i}' for i in range(cm.shape[0])])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.pdf'), bbox_inches='tight')
    plt.close()