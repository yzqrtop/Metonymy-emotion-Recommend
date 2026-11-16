import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
from typing import Dict, List, Optional, Tuple
import wandb
from transformers import get_linear_schedule_with_warmup

from src.models.metonymy_rec import MetonymyRec

class MetonymyDataset(Dataset):
    """
    转喻感知推荐数据集
    数据格式参照Amazon Reviews 2023[citation:10]
    """
    
    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 50,
        text_dim: int = 768,
        image_dim: int = 1024
    ):
        self.data = pd.read_parquet(data_path)
        self.max_seq_len = max_seq_len
        self.text_dim = text_dim
        self.image_dim = image_dim
        
        # 构建物品ID映射
        self.item2id = {item: i+1 for i, item in enumerate(self.data['item_id'].unique())}
        self.id2item = {i+1: item for i, item in enumerate(self.data['item_id'].unique())}
        
        # 按用户分组
        self.user_data = self._group_by_user()
        
    def _group_by_user(self) -> Dict:
        """按用户ID分组"""
        user_groups = {}
        for user_id, group in self.data.groupby('user_id'):
            # 按时间排序
            group = group.sort_values('timestamp')
            
            # 提取特征
            items = group['item_id'].map(self.item2id).tolist()
            text_feats = group['text_feature'].tolist()  # 假设已预计算
            image_feats = group['image_feature'].tolist()
            ratings = group['rating'].tolist()
            
            # 转喻标签
            if 'metonymy_type' in group.columns:
                metonymy_labels = group['metonymy_type'].tolist()
            else:
                metonymy_labels = [None] * len(group)
                
            user_groups[user_id] = {
                'items': items,
                'text_features': text_feats,
                'image_features': image_feats,
                'ratings': ratings,
                'metonymy_labels': metonymy_labels,
                'length': len(items)
            }
            
        return user_groups
    
    def __len__(self):
        return len(self.user_data)
    
    def __getitem__(self, idx):
        user_id = list(self.user_data.keys())[idx]
        user_seq = self.user_data[user_id]
        
        # 随机选择一个分割点
        if user_seq['length'] <= self.max_seq_len + 1:
            seq_len = user_seq['length'] - 1
            start_idx = 0
        else:
            seq_len = self.max_seq_len
            start_idx = np.random.randint(0, user_seq['length'] - seq_len - 1)
        
        # 历史序列
        items = user_seq['items'][start_idx:start_idx + seq_len]
        text_feats = user_seq['text_features'][start_idx:start_idx + seq_len]
        image_feats = user_seq['image_features'][start_idx:start_idx + seq_len]
        
        # 目标物品（下一个）
        target_item = user_seq['items'][start_idx + seq_len]
        target_rating = user_seq['ratings'][start_idx + seq_len]
        
        # 转换为二分类标签（4星以上为正样本）
        label = 1.0 if target_rating >= 4 else 0.0
        
        # 填充到固定长度
        pad_len = self.max_seq_len - seq_len
        if pad_len > 0:
            items = [0] * pad_len + items
            text_feats = [np.zeros(self.text_dim)] * pad_len + text_feats
            image_feats = [np.zeros(self.image_dim)] * pad_len + image_feats
            mask = [0] * pad_len + [1] * seq_len
        else:
            mask = [1] * self.max_seq_len
            
        return {
            'item_seq': torch.tensor(items, dtype=torch.long),
            'text_features': torch.tensor(np.array(text_feats), dtype=torch.float32),
            'image_features': torch.tensor(np.array(image_feats), dtype=torch.float32),
            'target_item': torch.tensor(target_item, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32)
        }


class MetonymyTrainer:
    """
    训练器
    """
    
    def __init__(
        self,
        model: MetonymyRec,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        device: str = 'cuda',
        use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="metonymy-rec")
            
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_main_loss = 0
        total_div_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # 移动到设备
            item_seq = batch['item_seq'].to(self.device)
            text_features = batch['text_features'].to(self.device)
            image_features = batch['image_features'].to(self.device)
            target_item = batch['target_item'].to(self.device)
            labels = batch['label'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # 前向传播
            outputs = self.model(
                item_seq,
                text_features,
                image_features,
                None,  # 暂时没有转喻字幕
                target_item,
                mask
            )
            
            # 计算损失
            loss = self.model.compute_loss(
                outputs['logits'],
                labels,
                outputs['diversity_loss'],
                alpha=0.01
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            total_main_loss += F.binary_cross_entropy_with_logits(
                outputs['logits'], labels
            ).item()
            total_div_loss += outputs['diversity_loss'].item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'main_loss': total_main_loss / (pbar.n + 1),
                'div_loss': total_div_loss / (pbar.n + 1)
            })
            
        return {
            'loss': total_loss / len(self.train_loader),
            'main_loss': total_main_loss / len(self.train_loader),
            'div_loss': total_div_loss / len(self.train_loader)
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        all_labels = []
        all_preds = []
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            item_seq = batch['item_seq'].to(self.device)
            text_features = batch['text_features'].to(self.device)
            image_features = batch['image_features'].to(self.device)
            target_item = batch['target_item'].to(self.device)
            labels = batch['label'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            outputs = self.model(
                item_seq,
                text_features,
                image_features,
                None,
                target_item,
                mask
            )
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs['logits']).cpu().numpy())
        
        # 计算指标
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
        
        return {
            'auc': auc,
            'accuracy': acc,
            'f1': f1
        }
    
    def train(self, epochs: int, save_path: str):
        """完整训练流程"""
        best_auc = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.evaluate()
            
            # 打印
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            # 记录
            if self.use_wandb:
                wandb.log({
                    'train_loss': train_metrics['loss'],
                    'train_main_loss': train_metrics['main_loss'],
                    'train_div_loss': train_metrics['div_loss'],
                    'val_auc': val_metrics['auc'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1'],
                    'epoch': epoch
                })
            
            # 保存最佳模型
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_auc': best_auc
                }, f"{save_path}/best_model.pt")
                print(f"Saved best model with AUC: {best_auc:.4f}")


# 主训练脚本
if __name__ == "__main__":
    # 配置
    config = {
        'data_path': './data/train.parquet',
        'val_path': './data/val.parquet',
        'batch_size': 64,
        'max_seq_len': 50,
        'hidden_dim': 256,
        'text_dim': 768,
        'image_dim': 1024,
        'n_prototypes': 16,
        'n_heads': 4,
        'n_layers': 2,
        'lr': 1e-4,
        'epochs': 20,
        'save_path': './checkpoints',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 创建数据集
    train_dataset = MetonymyDataset(
        config['data_path'],
        max_seq_len=config['max_seq_len'],
        text_dim=config['text_dim'],
        image_dim=config['image_dim']
    )
    
    val_dataset = MetonymyDataset(
        config['val_path'],
        max_seq_len=config['max_seq_len'],
        text_dim=config['text_dim'],
        image_dim=config['image_dim']
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    model = MetonymyRec(
        item_num=len(train_dataset.item2id) + 1,
        text_dim=config['text_dim'],
        image_dim=config['image_dim'],
        hidden_dim=config['hidden_dim'],
        max_seq_len=config['max_seq_len'],
        n_prototypes=config['n_prototypes'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers']
    )
    
    # 创建训练器
    trainer = MetonymyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config['lr'],
        device=config['device'],
        use_wandb=True
    )
    
    # 创建保存目录
    os.makedirs(config['save_path'], exist_ok=True)
    
    # 开始训练
    trainer.train(
        epochs=config['epochs'],
        save_path=config['save_path']
    )