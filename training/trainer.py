#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练器模块
包含完整的训练流程
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score

from models import SpectralNet
from utils.metrics import calculate_metrics


def train_spectral_model(args, train_loader, val_loader, test_loader):
    """训练光谱模型"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 创建模型
    config = {
        'feature_dim': args.feature_dim,
        'dropout_rate': args.dropout_rate,
        'spectral_attention_reduction': args.spectral_attention_reduction,
        'spatial_attention_heads': args.spatial_attention_heads
    }
    
    model = SpectralNet(
        num_bands=args.num_bands,
        num_classes=args.num_classes,
        config=config
    ).to(device)
    
    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"模型参数: 总数={total_params:,}, 可训练={trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=args.epochs // 2,
        T_mult=2, 
        eta_min=args.lr * 0.01
    )
    
    # 混合精度
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    logging.info("开始训练...")
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch
        )
        
        # 验证阶段
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1_score'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # 更新学习率
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        
        # 日志输出
        logging.info(
            f"Epoch {epoch+1:3d}: "
            f"Train - Loss:{train_loss:.4f}, Acc:{train_metrics['accuracy']:.4f}, F1:{train_metrics['f1_score']:.4f} | "
            f"Val - Loss:{val_loss:.4f}, Acc:{val_metrics['accuracy']:.4f}, F1:{val_metrics['f1_score']:.4f} | "
            f"LR:{optimizer.param_groups[0]['lr']:.6f}, Time:{epoch_time:.1f}s"
        )
        
        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config,
                'history': history
            }, os.path.join(args.output_dir, 'best_spectral_model.pt'))
            
            logging.info(f"新的最佳验证准确率: {best_val_acc:.4f}")
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= args.patience:
            logging.info(f"早停触发，第 {epoch+1} 轮")
            break
    
    # 加载最佳模型进行测试
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_spectral_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试评估
    test_loss, test_metrics, test_targets, test_preds = evaluate_model(
        model, test_loader, criterion, device, args.num_classes
    )
    
    # 打印最终结果
    logging.info("=" * 80)
    logging.info("最终测试结果:")
    logging.info("=" * 80)
    logging.info(f"测试准确率: {test_metrics['accuracy']:.4f}")
    logging.info(f"测试F1分数: {test_metrics['f1_score']:.4f}")
    logging.info(f"测试Kappa系数: {test_metrics['kappa']:.4f}")
    logging.info(f"测试MCC系数: {test_metrics['mcc']:.4f}")
    logging.info("=" * 80)
    
    return model, history, best_val_acc, test_metrics, test_targets, test_preds


def train_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch):
    """训练一个epoch"""
    model.train()
    train_loss = 0.0
    train_targets = []
    train_preds = []
    
    train_pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}", leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(train_pbar):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler:
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # 收集训练数据
        with torch.no_grad():
            _, predicted = outputs.max(1)
            train_targets.extend(targets.cpu().numpy())
            train_preds.extend(predicted.cpu().numpy())
        
        train_loss += loss.item()
        
        # 实时显示
        if len(train_targets) > 0:
            current_acc = accuracy_score(train_targets, train_preds)
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{current_acc:.4f}"
            })
    
    train_loss = train_loss / len(train_loader)
    train_metrics = calculate_metrics(train_targets, train_preds, model.num_classes)
    
    return train_loss, train_metrics


def validate_epoch(model, val_loader, criterion, device, epoch):
    """验证一个epoch"""
    model.eval()
    val_loss = 0.0
    val_targets = []
    val_preds = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f"验证 Epoch {epoch+1}", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            
            _, predicted = outputs.max(1)
            val_targets.extend(targets.cpu().numpy())
            val_preds.extend(predicted.cpu().numpy())
    
    val_loss = val_loss / len(val_loader)
    val_metrics = calculate_metrics(val_targets, val_preds, model.num_classes)
    
    return val_loss, val_metrics


def evaluate_model(model, test_loader, criterion, device, num_classes):
    """测试模型"""
    model.eval()
    test_loss = 0.0
    test_targets = []
    test_preds = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="测试评估", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            test_targets.extend(targets.cpu().numpy())
            test_preds.extend(predicted.cpu().numpy())
    
    test_loss = test_loss / len(test_loader)
    test_metrics = calculate_metrics(test_targets, test_preds, num_classes)
    
    return test_loss, test_metrics, test_targets, test_preds