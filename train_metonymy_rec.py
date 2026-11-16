import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.metonymy_rec import MetonymyRec


class YelpMetonymyDataset(Dataset):
    """
    Yelp数据集 - 转喻感知推荐数据集
    从photos_with_metonymy_annotations.json加载数据
    
    数据特点：每个商家只有一张照片，因此采用以下策略：
    1. 将label作为类别，构建类别-物品的映射
    2. 使用转喻特征增强表示
    3. 将问题转化为多类别分类或相似度学习
    
    支持训练/验证/测试集划分
    """
    
    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 5,
        text_dim: int = 768,
        image_dim: int = 1024,
        base_image_dir: str = None,
        split: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ):
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.base_image_dir = base_image_dir or os.path.dirname(data_path)
        self.split = split
        
        # 数据划分比例
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "划分比例之和必须为1"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # 加载数据
        self.raw_data = self._load_data()
        
        # 构建物品ID映射 (使用photo_id作为物品ID)
        self.item2id = {item: i+1 for i, item in enumerate(self.raw_data['photo_id'].unique())}
        self.id2item = {i+1: item for i, item in enumerate(self.raw_data['photo_id'].unique())}
        
        # 构建类别映射 (使用label作为类别)
        self.label2id = {label: i for i, label in enumerate(self.raw_data['label'].unique())}
        self.id2label = {i: label for i, label in enumerate(self.raw_data['label'].unique())}
        
        # 按类别分组构建序列
        self.label_sequences = self._build_label_sequences()
        
        # 创建训练样本
        self.samples = self._create_samples()
        
        # 数据划分
        self.train_samples, self.val_samples, self.test_samples = self._split_samples()
        
        # 根据split参数选择样本
        if split == 'train':
            self.split_samples = self.train_samples
        elif split == 'val':
            self.split_samples = self.val_samples
        else:  # 'test'
            self.split_samples = self.test_samples
        
    def _load_data(self):
        """加载JSON Lines格式的数据"""
        import pandas as pd
        
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    data.append(record)
                except json.JSONDecodeError:
                    continue
        
        df = pd.DataFrame(data)
        
        # 解析metonymy_annotation字段
        if 'metonymy_annotation' in df.columns:
            annotation_df = pd.json_normalize(df['metonymy_annotation'])
            df = pd.concat([df.drop('metonymy_annotation', axis=1), annotation_df], axis=1)
        
        return df
    
    def _build_label_sequences(self):
        """按类别构建序列"""
        label_sequences = {}
        
        # 按label分组
        for label, group in self.raw_data.groupby('label'):
            label_id = self.label2id[label]
            
            # 为每个类别创建物品序列
            items = []
            text_feats = []
            image_feats = []
            labels = []
            business_ids = []
            
            for _, row in group.iterrows():
                photo_id = row['photo_id']
                if photo_id in self.item2id:
                    items.append(self.item2id[photo_id])
                    
                    # 生成文本特征（使用caption的embedding）
                    caption = row.get('caption', '')
                    text_feat = self._text_to_feature(caption)
                    text_feats.append(text_feat)
                    
                    # 生成图像特征（使用占位符）
                    image_feat = self._image_to_feature(row.get('image_path', ''))
                    image_feats.append(image_feat)
                    
                    # 标签：基于sentiment
                    sentiment = row.get('inferred_sentiment', 'neutral')
                    label_val = 1.0 if sentiment == 'positive' else 0.0 if sentiment == 'negative' else 0.5
                    labels.append(label_val)
                    
                    business_ids.append(row.get('business_id', ''))
            
            if len(items) > 0:
                label_sequences[label_id] = {
                    'items': items,
                    'text_features': text_feats,
                    'image_features': image_feats,
                    'labels': labels,
                    'business_ids': business_ids,
                    'length': len(items),
                    'label_name': label
                }
        
        return label_sequences
    
    def _text_to_feature(self, text: str) -> np.ndarray:
        """将文本转换为特征向量（简化版本，使用哈希）"""
        if not text:
            return np.zeros(self.text_dim)
        
        # 使用文本哈希生成特征
        hash_val = hash(text) % (2**32)
        np.random.seed(hash_val)
        feature = np.random.randn(self.text_dim).astype(np.float32)
        # 归一化
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        return feature * 0.1
    
    def _image_to_feature(self, image_path: str) -> np.ndarray:
        """将图像转换为特征向量（简化版本）"""
        np.random.seed(hash(image_path) % (2**32))
        feature = np.random.randn(self.image_dim).astype(np.float32)
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        return feature * 0.1
    
    def _create_samples(self):
        """创建训练样本"""
        samples = []
        
        sentiment2id = {'positive': 0, 'neutral': 1, 'negative': 2}
        metonymy_type2id = {'T1': 0, 'T2': 1, 'T3': 2, 'null': 3}
        
        # 为每个类别内的物品创建序列样本
        for label_id, seq_data in self.label_sequences.items():
            items = seq_data['items']
            text_feats = seq_data['text_features']
            image_feats = seq_data['image_features']
            labels = seq_data['labels']
            length = seq_data['length']
            
            # 如果类别内物品数量足够，创建滑动窗口样本
            if length >= 2:
                window_size = min(self.max_seq_len, length - 1)
                
                for i in range(window_size, length):
                    # 历史序列
                    start_idx = max(0, i - self.max_seq_len)
                    hist_items = items[start_idx:i]
                    hist_text = text_feats[start_idx:i]
                    hist_image = image_feats[start_idx:i]
                    
                    # 目标物品
                    target_item = items[i]
                    target_label = labels[i]
                    
                    # 获取转喻信息
                    target_photo_id = self.id2item.get(target_item, '')
                    sentiment_id, metonymy_type_id = self._get_metonymy_info(target_photo_id)
                    
                    samples.append({
                        'label_id': label_id,
                        'hist_items': hist_items,
                        'hist_text': hist_text,
                        'hist_image': hist_image,
                        'target_item': target_item,
                        'label': target_label,
                        'sentiment_id': sentiment_id,
                        'metonymy_type_id': metonymy_type_id,
                        'photo_id': target_photo_id
                    })
            else:
                # 如果只有一个物品，创建单物品样本
                target_photo_id = self.id2item.get(items[0], '')
                sentiment_id, metonymy_type_id = self._get_metonymy_info(target_photo_id)
                
                samples.append({
                    'label_id': label_id,
                    'hist_items': [items[0]],
                    'hist_text': [text_feats[0]],
                    'hist_image': [image_feats[0]],
                    'target_item': items[0],
                    'label': labels[0],
                    'sentiment_id': sentiment_id,
                    'metonymy_type_id': metonymy_type_id,
                    'photo_id': target_photo_id
                })
        
        # 如果样本太少，添加一些跨类别的对比样本
        if len(samples) < 10:
            samples = self._create_contrastive_samples(samples)
        
        return samples
    
    def _get_metonymy_info(self, photo_id: str):
        """获取照片的转喻信息"""
        sentiment2id = {'positive': 0, 'neutral': 1, 'negative': 2}
        metonymy_type2id = {'T1': 0, 'T2': 1, 'T3': 2, 'null': 3}
        
        for _, row in self.raw_data.iterrows():
            if row.get('photo_id') == photo_id:
                sentiment = row.get('inferred_sentiment', 'neutral')
                metonymy_type = row.get('metonymy_type', 'null')
                return sentiment2id.get(sentiment, 1), metonymy_type2id.get(metonymy_type, 3)
        return 1, 3  # 默认返回neutral和null
    
    def _split_samples(self):
        """划分训练/验证/测试集"""
        samples = self.samples
        np.random.seed(42)  # 固定随机种子
        indices = np.random.permutation(len(samples))
        
        n_total = len(samples)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]
        test_samples = [samples[i] for i in test_indices]
        
        print(f"数据划分: 训练集 {len(train_samples)}, 验证集 {len(val_samples)}, 测试集 {len(test_samples)}")
        
        return train_samples, val_samples, test_samples
    
    def _create_contrastive_samples(self, existing_samples):
        """创建对比学习样本"""
        all_items = list(self.item2id.values())
        
        # 为每个物品创建正样本和负样本
        for label_id, seq_data in self.label_sequences.items():
            items = seq_data['items']
            text_feats = seq_data['text_features']
            image_feats = seq_data['image_features']
            
            for i, item in enumerate(items):
                # 正样本（同类别）
                if len(items) > 1:
                    other_idx = (i + 1) % len(items)
                    existing_samples.append({
                        'label_id': label_id,
                        'hist_items': [item],
                        'hist_text': [text_feats[i]],
                        'hist_image': [image_feats[i]],
                        'target_item': items[other_idx],
                        'label': 1.0  # 正样本
                    })
                
                # 负样本（不同类别）
                other_label_id = (label_id + 1) % len(self.label_sequences)
                if other_label_id in self.label_sequences:
                    other_items = self.label_sequences[other_label_id]['items']
                    if other_items:
                        other_item = other_items[0]
                        other_text = self.label_sequences[other_label_id]['text_features'][0]
                        other_image = self.label_sequences[other_label_id]['image_features'][0]
                        
                        existing_samples.append({
                            'label_id': label_id,
                            'hist_items': [item],
                            'hist_text': [text_feats[i]],
                            'hist_image': [image_feats[i]],
                            'target_item': other_item,
                            'label': 0.0  # 负样本
                        })
        
        return existing_samples
    
    def __len__(self):
        return len(self.split_samples)
    
    def __getitem__(self, idx):
        sample = self.split_samples[idx]
        
        hist_items = sample['hist_items']
        hist_text = sample['hist_text']
        hist_image = sample['hist_image']
        
        seq_len = len(hist_items)
        
        # 填充或截断到固定长度
        if seq_len > self.max_seq_len:
            hist_items = hist_items[-self.max_seq_len:]
            hist_text = hist_text[-self.max_seq_len:]
            hist_image = hist_image[-self.max_seq_len:]
            seq_len = self.max_seq_len
        
        pad_len = self.max_seq_len - seq_len
        if pad_len > 0:
            hist_items = [0] * pad_len + hist_items
            hist_text = [np.zeros(self.text_dim)] * pad_len + hist_text
            hist_image = [np.zeros(self.image_dim)] * pad_len + hist_image
            mask = [0] * pad_len + [1] * seq_len
        else:
            mask = [1] * self.max_seq_len
        
        return {
            'item_seq': torch.tensor(hist_items, dtype=torch.long),
            'text_features': torch.tensor(np.array(hist_text), dtype=torch.float32),
            'image_features': torch.tensor(np.array(hist_image), dtype=torch.float32),
            'target_item': torch.tensor(sample['target_item'], dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'sentiment_id': torch.tensor(sample.get('sentiment_id', 1), dtype=torch.long),
            'metonymy_type_id': torch.tensor(sample.get('metonymy_type_id', 3), dtype=torch.long),
            'photo_id': sample.get('photo_id', '')
        }


class MetonymyTrainer:
    """训练器"""
    
    def __init__(
        self,
        model: MetonymyRec,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        dataset=None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = 'cuda',
        train_config: dict = None,
        patience: int = 5,  # 早停耐心值
        l1_lambda: float = 0.0001  # L1正则化系数
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset = dataset
        self.device = device
        self.train_config = train_config or {}
        self.patience = patience
        self.l1_lambda = l1_lambda
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'batch_loss': []  # 记录每个批次的损失
        }
        
        # 早停相关
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.early_stopped = False
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',  # 基于验证损失
            factor=0.5,  # 学习率减少因子
            patience=3,  # 耐心值
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=1e-6,  # 最小学习率
            eps=1e-08
        )
        
        # TensorBoard 集成
        log_dir = os.path.join(os.path.dirname(self.train_config.get('save_path', './')), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard 日志保存到: {log_dir}")
        print(f"使用命令查看: tensorboard --logdir={log_dir}")
        
    def train_epoch(self) -> dict:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        current_epoch = len(self.history['train_loss']) + 1
        global_step = current_epoch * len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
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
                target_item,
                None,
                mask
            )
            
            # 计算损失（添加L1正则化）
            loss = self.model.compute_loss(
                outputs['logits'],
                labels,
                outputs['diversity_loss'],
                alpha=0.01,
                l1_lambda=self.l1_lambda
            )
            
            # 计算当前步骤
            step = (current_epoch - 1) * len(self.train_loader) + batch_idx
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 记录模型参数的梯度与分布到 TensorBoard
            if batch_idx % 100 == 0:  # 每100个批次记录一次
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        # 记录梯度直方图
                        self.writer.add_histogram(f'Gradients/{name}', param.grad, step)
                        # 记录参数直方图
                        self.writer.add_histogram(f'Parameters/{name}', param, step)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            num_batches += 1
            
            # 记录批次损失到 TensorBoard
            self.writer.add_scalar('Batch_Loss', loss.item(), step)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], step)
            
            # 记录批次损失到历史
            self.history['batch_loss'].append(loss.item())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # 记录训练损失到 TensorBoard
        self.writer.add_scalar('Epoch_Loss/Train', avg_loss, current_epoch)
        
        # 记录训练损失
        self.history['train_loss'].append(avg_loss)
        
        return {
            'loss': avg_loss
        }

    def validate(self, k_list=[1, 5, 10]) -> dict:
        """
        验证模型性能，使用推荐系统评估指标

        Args:
            k_list: 评估K值列表

        Returns:
            验证指标字典
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0

        current_epoch = len(self.history['train_loss'])

        # 初始化推荐系统评估指标
        hr_metrics = {f'HR@{k}': 0 for k in k_list}
        ndcg_metrics = {f'NDCG@{k}': 0 for k in k_list}
        recall_metrics = {f'Recall@{k}': 0 for k in k_list}
        accuracy = 0  # 准确率（使用HR@1）
        total_tests = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                item_seq = batch['item_seq'].to(self.device)
                text_features = batch['text_features'].to(self.device)
                image_features = batch['image_features'].to(self.device)
                target_item = batch['target_item'].to(self.device)
                labels = batch['label'].to(self.device)
                mask = batch['mask'].to(self.device)

                # 计算损失
                outputs = self.model(
                    item_seq,
                    text_features,
                    image_features,
                    target_item,
                    None,
                    mask
                )

                loss = self.model.compute_loss(
                    outputs['logits'],
                    labels,
                    outputs['diversity_loss'],
                    alpha=0.01,
                    l1_lambda=self.l1_lambda
                )

                total_loss += loss.item()

                # 计算推荐系统指标
                batch_size = item_seq.size(0)
                for i in range(batch_size):
                    query_id = item_seq[i, -1].item()
                    target_id = target_item[i].item()

                    # 获取该批次中所有物品的得分（批量处理）
                    all_item_ids = [item_id for item_id in range(1, len(self.dataset.item2id) + 1) if item_id != query_id]
                    all_scores = []
                    
                    # 批量处理参数
                    batch_size_items = 64
                    item_seq_tensor = item_seq[i:i+1]
                    text_tensor = text_features[i:i+1]
                    image_tensor = image_features[i:i+1]
                    mask_tensor = mask[i:i+1]
                    
                    # 批量预测
                    for j in range(0, len(all_item_ids), batch_size_items):
                        batch_item_ids = all_item_ids[j:j+batch_size_items]
                        if not batch_item_ids:
                            continue
                        
                        # 创建批次目标张量
                        batch_targets = torch.tensor(batch_item_ids, dtype=torch.long).to(self.device)
                        
                        # 重复序列特征以匹配批次大小
                        batch_seq = item_seq_tensor.repeat(len(batch_item_ids), 1)
                        batch_text = text_tensor.repeat(len(batch_item_ids), 1, 1)
                        batch_image = image_tensor.repeat(len(batch_item_ids), 1, 1)
                        batch_mask = mask_tensor.repeat(len(batch_item_ids), 1)
                        
                        # 一次前向传播处理整个批次
                        out = self.model(
                            batch_seq,
                            batch_text,
                            batch_image,
                            batch_targets,
                            None,
                            batch_mask
                        )
                        scores = torch.sigmoid(out['logits']).cpu().numpy()
                        
                        # 收集结果
                        for item_id, score in zip(batch_item_ids, scores):
                            all_scores.append((item_id, score))

                    # 排序
                    all_scores.sort(key=lambda x: x[1], reverse=True)

                    # 计算各K值的指标
                    for k in k_list:
                        top_k = [item_id for item_id, _ in all_scores[:k]]

                        # HR@k
                        if target_id in top_k:
                            hr_metrics[f'HR@{k}'] += 1

                        # NDCG@k
                        if target_id in top_k:
                            rank = top_k.index(target_id) + 1
                            ndcg = 1 / np.log2(rank + 1)
                            ndcg_metrics[f'NDCG@{k}'] += ndcg

                        # Recall@k
                        if target_id in top_k:
                            recall_metrics[f'Recall@{k}'] += 1

                        # 准确率（HR@1）
                        if k == 1 and target_id == top_k[0]:
                            accuracy += 1

                    total_tests += 1

        # 计算平均指标
        val_loss = total_loss / len(self.val_loader)

        # 计算最终指标
        if total_tests > 0:
            for k in k_list:
                hr_metrics[f'HR@{k}'] /= total_tests
                ndcg_metrics[f'NDCG@{k}'] /= total_tests
                recall_metrics[f'Recall@{k}'] /= total_tests
            accuracy = accuracy / total_tests  # 计算平均准确率

        # 记录到历史
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(accuracy)  # 记录验证准确率
        for k in k_list:
            if f'HR@{k}' not in self.history:
                self.history[f'HR@{k}'] = []
            self.history[f'HR@{k}'].append(hr_metrics[f'HR@{k}'])
            if f'NDCG@{k}' not in self.history:
                self.history[f'NDCG@{k}'] = []
            self.history[f'NDCG@{k}'].append(ndcg_metrics[f'NDCG@{k}'])
            if f'Recall@{k}' not in self.history:
                self.history[f'Recall@{k}'] = []
            self.history[f'Recall@{k}'].append(recall_metrics[f'Recall@{k}'])

        # 记录到 TensorBoard
        if current_epoch > 0:
            self.writer.add_scalar('Epoch_Loss/Validation', val_loss, current_epoch)
            self.writer.add_scalar('Accuracy/Validation', accuracy, current_epoch)  # 记录准确率
            for k in k_list:
                self.writer.add_scalar(f'HR@{k}/Validation', hr_metrics[f'HR@{k}'], current_epoch)
                self.writer.add_scalar(f'NDCG@{k}/Validation', ndcg_metrics[f'NDCG@{k}'], current_epoch)
                self.writer.add_scalar(f'Recall@{k}/Validation', recall_metrics[f'Recall@{k}'], current_epoch)

        return {
            'loss': val_loss,
            'accuracy': accuracy,  # 添加准确率
            'hr': hr_metrics,
            'ndcg': ndcg_metrics,
            'recall': recall_metrics,
            'total_tests': total_tests
        }
    
    def train(self, epochs: int, save_path: str):
        """训练模型"""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # 训练
            train_metrics = self.train_epoch()
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            
            # 验证
            if self.val_loader is not None:
                val_metrics = self.validate()
                hr_info = ', '.join([f"HR@{k}={v:.4f}" for k, v in val_metrics['hr'].items()])
                ndcg_info = ', '.join([f"NDCG@{k}={v:.4f}" for k, v in val_metrics['ndcg'].items()])
                recall_info = ', '.join([f"Recall@{k}={v:.4f}" for k, v in val_metrics['recall'].items()])
                accuracy_info = f"Accuracy={val_metrics['accuracy']:.4f}"
                print(f"Val Loss: {val_metrics['loss']:.4f}")
                print(f"  {accuracy_info}")
                print(f"  {hr_info}")
                print(f"  {ndcg_info}")
                print(f"  {recall_info}")
                
                # 早停检查
                current_val_loss = val_metrics['loss']
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self.early_stop_counter = 0
                    # 保存最佳模型
                    best_path = os.path.join(save_path, "best_model.pt")
                    torch.save(self.model.state_dict(), best_path)
                    print(f"Saved best model to {best_path}")
                else:
                    self.early_stop_counter += 1
                    print(f"早停计数器: {self.early_stop_counter}/{self.patience}")
                    if self.early_stop_counter >= self.patience:
                        print("触发早停！停止训练以防止过拟合")
                        self.early_stopped = True
                        break
                
                # 更新学习率调度器
                self.scheduler.step(current_val_loss)
        
            # 保存检查点
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_metrics['loss'],
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        # 训练结束后生成可视化
        self.plot_training_history(save_path)
        self.save_training_summary(save_path)
        self.save_training_summary_markdown(save_path)
        
        # 关闭 TensorBoard writer
        self.writer.close()
    
    def plot_training_history(self, save_path: str):
        """绘制训练历史图表"""
        plt.figure(figsize=(15, 10))
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 训练和验证损失
        plt.subplot(2, 2, 1)
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='训练损失', linewidth=2)
        if self.history['val_loss']:
            plt.plot(epochs, self.history['val_loss'], 'r-', label='验证损失', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('训练和验证损失', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 2. HR@K 指标
        plt.subplot(2, 2, 2)
        for k in [5, 10]:
            if f'HR@{k}' in self.history and len(self.history[f'HR@{k}']) > 0:
                plt.plot(range(1, len(self.history[f'HR@{k}']) + 1), self.history[f'HR@{k}'], label=f'HR@{k}', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Hit Rate', fontsize=12)
        plt.title('HR@K 指标', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 3. 学习率变化
        plt.subplot(2, 2, 3)
        if self.history['learning_rate']:
            plt.plot(epochs, self.history['learning_rate'], 'm-', label='学习率', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.title('学习率变化', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
        
        # 4. 损失对比（训练 vs 验证）
        plt.subplot(2, 2, 4)
        if self.history['val_loss']:
            plt.plot(epochs, self.history['train_loss'], 'b-', label='训练损失', linewidth=2)
            plt.plot(epochs, self.history['val_loss'], 'r-', label='验证损失', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('训练 vs 验证损失对比', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(save_path, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图表已保存到: {plot_path}")
        plt.close()
    
    def save_training_summary(self, save_path: str):
        """保存训练总结文件"""
        summary = {
            'training_info': {
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_epochs': len(self.history['train_loss']),
                'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
                'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
                'final_val_accuracy': self.history['val_accuracy'][-1] if self.history['val_accuracy'] else None,
                'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else None,
                'best_val_accuracy': max(self.history['val_accuracy']) if self.history['val_accuracy'] else None,
                'initial_lr': self.history['learning_rate'][0] if self.history['learning_rate'] else None,
                'final_lr': self.history['learning_rate'][-1] if self.history['learning_rate'] else None,
            },
            'training_parameters': self.train_config,
            'training_history': self.history,
            'model_performance': {
                'train_loss_trend': 'decreasing' if len(self.history['train_loss']) > 1 and self.history['train_loss'][-1] < self.history['train_loss'][0] else 'stable',
                'val_loss_trend': 'decreasing' if len(self.history['val_loss']) > 1 and self.history['val_loss'][-1] < self.history['val_loss'][0] else 'stable',
                'overfitting_detected': len(self.history['train_loss']) > 5 and len(self.history['val_loss']) > 5 and 
                                       self.history['val_loss'][-1] > self.history['val_loss'][-5] and 
                                       self.history['train_loss'][-1] < self.history['train_loss'][-5],
            },
            'recommendations': []
        }
        
        # 生成建议
        if summary['model_performance']['overfitting_detected']:
            summary['recommendations'].append('检测到过拟合现象，建议增加数据增强、使用正则化或减少模型复杂度')
        
        if summary['training_info']['final_val_accuracy'] and summary['training_info']['final_val_accuracy'] < 0.7:
            summary['recommendations'].append('验证准确率较低，建议增加训练数据、调整学习率或优化模型结构')
        
        if len(summary['recommendations']) == 0:
            summary['recommendations'].append('训练过程正常，模型性能良好')
        
        # 保存为JSON文件
        summary_path = os.path.join(save_path, 'training_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"训练总结已保存到: {summary_path}")
        
        # 生成可读的文本总结
        text_summary_path = os.path.join(save_path, 'training_summary.txt')
        with open(text_summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MetonymyRec 模型训练总结\n")
            f.write("=" * 80 + "\n\n")
            
            def safe_format(val, fmt='.4f'):
                """安全格式化数值，处理None值"""
                if val is None:
                    return 'N/A'
                return f'{val:{fmt}}'
            
            f.write("【训练信息】\n")
            f.write("-" * 80 + "\n")
            f.write(f"开始时间: {summary['training_info']['start_time']}\n")
            f.write(f"训练轮数: {summary['training_info']['total_epochs']}\n")
            f.write(f"最终训练损失: {safe_format(summary['training_info']['final_train_loss'])}\n")
            f.write(f"最终验证损失: {safe_format(summary['training_info']['final_val_loss'])}\n")
            f.write(f"最终验证准确率: {safe_format(summary['training_info']['final_val_accuracy'])}\n")
            f.write(f"最佳验证损失: {safe_format(summary['training_info']['best_val_loss'])}\n")
            f.write(f"最佳验证准确率: {safe_format(summary['training_info']['best_val_accuracy'])}\n")
            f.write(f"初始学习率: {safe_format(summary['training_info']['initial_lr'], '.6f')}\n")
            f.write(f"最终学习率: {safe_format(summary['training_info']['final_lr'], '.6f')}\n\n")
            
            f.write("【训练参数】\n")
            f.write("-" * 80 + "\n")
            for key, value in summary['training_parameters'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("【模型性能分析】\n")
            f.write("-" * 80 + "\n")
            f.write(f"训练损失趋势: {summary['model_performance']['train_loss_trend']}\n")
            f.write(f"验证损失趋势: {summary['model_performance']['val_loss_trend']}\n")
            f.write(f"过拟合检测: {'是' if summary['model_performance']['overfitting_detected'] else '否'}\n\n")
            
            f.write("【训练建议】\n")
            f.write("-" * 80 + "\n")
            for i, rec in enumerate(summary['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            
            f.write("【训练历史详情】\n")
            f.write("-" * 80 + "\n")
            f.write("Epoch\tTrain Loss\tVal Loss\tVal Acc\tLearning Rate\n")
            for i in range(len(self.history['train_loss'])):
                train_loss = self.history['train_loss'][i]
                val_loss = self.history['val_loss'][i] if i < len(self.history['val_loss']) else 'N/A'
                val_acc = self.history['val_accuracy'][i] if i < len(self.history['val_accuracy']) else 'N/A'
                lr = self.history['learning_rate'][i] if i < len(self.history['learning_rate']) else 'N/A'
                f.write(f"{i+1}\t{safe_format(train_loss)}\t\t{val_loss}\t\t{val_acc}\t\t{lr}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("训练总结生成完成\n")
            f.write("=" * 80 + "\n")
        
        print(f"可读训练总结已保存到: {text_summary_path}")
    
    def save_training_summary_markdown(self, save_path: str):
        """保存 Markdown 格式的训练总结文件"""
        summary = {
            'training_info': {
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_epochs': len(self.history['train_loss']),
                'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
                'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
                'final_val_accuracy': self.history['val_accuracy'][-1] if self.history['val_accuracy'] else None,
                'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else None,
                'best_val_accuracy': max(self.history['val_accuracy']) if self.history['val_accuracy'] else None,
                'initial_lr': self.history['learning_rate'][0] if self.history['learning_rate'] else None,
                'final_lr': self.history['learning_rate'][-1] if self.history['learning_rate'] else None,
                'batch_size': self.train_config.get('batch_size', 'N/A'),
                'dataset_size': self.train_config.get('dataset_size', 'N/A'),
                'device': self.device,
            },
            'training_parameters': self.train_config,
            'training_history': self.history,
            'model_performance': {
                'train_loss_trend': 'decreasing' if len(self.history['train_loss']) > 1 and self.history['train_loss'][-1] < self.history['train_loss'][0] else 'stable',
                'val_loss_trend': 'decreasing' if len(self.history['val_loss']) > 1 and self.history['val_loss'][-1] < self.history['val_loss'][0] else 'stable',
                'overfitting_detected': len(self.history['train_loss']) > 5 and len(self.history['val_loss']) > 5 and 
                                       self.history['val_loss'][-1] > self.history['val_loss'][-5] and 
                                       self.history['train_loss'][-1] < self.history['train_loss'][-5],
                'converged': len(self.history['train_loss']) > 3 and abs(self.history['train_loss'][-1] - self.history['train_loss'][-3]) < 0.01,
            },
            'recommendations': []
        }
        
        # 生成建议
        if summary['model_performance']['overfitting_detected']:
            summary['recommendations'].append('检测到过拟合现象，建议增加数据增强、使用正则化或减少模型复杂度')
        
        if summary['training_info']['final_val_accuracy'] and summary['training_info']['final_val_accuracy'] < 0.7:
            summary['recommendations'].append('验证准确率较低，建议增加训练数据、调整学习率或优化模型结构')
        
        if not summary['model_performance']['converged']:
            summary['recommendations'].append('模型可能未完全收敛，建议增加训练轮数或调整学习率')
        
        if len(summary['recommendations']) == 0:
            summary['recommendations'].append('训练过程正常，模型性能良好')
        
        def safe_format(val, fmt='.4f'):
            """安全格式化数值，处理None值"""
            if val is None:
                return 'N/A'
            return f'{val:{fmt}}'
        
        # 保存为 Markdown 文件
        md_path = os.path.join(save_path, 'training_summary.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('# MetonymyRec 模型训练总结\n\n')
            
            # 一、结构化信息归档
            f.write('## 一、结构化信息归档\n\n')
            
            # 1. 基础信息块
            f.write('### 1. 基础信息块\n\n')
            f.write('| 项目 | 值 |\n')
            f.write('|------|-----|\n')
            f.write(f'| 训练开始时间 | {summary["training_info"]["start_time"]} |\n')
            f.write(f'| 总训练轮数 | {summary["training_info"]["total_epochs"]} |\n')
            f.write(f'| 学习率 | {safe_format(summary["training_info"]["initial_lr"], ".6f")} → {safe_format(summary["training_info"]["final_lr"], ".6f")} |\n')
            f.write(f'| 批次大小 | {summary["training_info"]["batch_size"]} |\n')
            f.write(f'| 使用设备 | {summary["training_info"]["device"]} |\n')
            f.write(f'| 数据集规模 | {summary["training_info"]["dataset_size"]} |\n\n')
            
            # 2. 性能指标块
            f.write('### 2. 性能指标块\n\n')
            f.write('| 指标 | 值 |\n')
            f.write('|------|-----|\n')
            f.write(f'| 最终训练损失 | {safe_format(summary["training_info"]["final_train_loss"])} |\n')
            f.write(f'| 最终验证损失 | {safe_format(summary["training_info"]["final_val_loss"])} |\n')
            f.write(f'| 最终验证准确率 | {safe_format(summary["training_info"]["final_val_accuracy"])} |\n')
            f.write(f'| 最佳验证损失 | {safe_format(summary["training_info"]["best_val_loss"])} |\n')
            f.write(f'| 最佳验证准确率 | {safe_format(summary["training_info"]["best_val_accuracy"])} |\n\n')
            
            # 3. 场景性能块
            f.write('### 3. 场景性能块\n\n')
            f.write('| 场景 | 表现 | 状态 |\n')
            f.write('|------|-----|------|\n')
            
            # 运行评估获取场景性能指标
            try:
                if hasattr(self, 'evaluate'):
                    eval_results = self.evaluate()
                    
                    # 长尾物品性能
                    if 'long_tail_performance' in eval_results and eval_results['long_tail_performance']:
                        long_tail_hr1 = eval_results['long_tail_performance'].get('HR@1', 0)
                        long_tail_hr5 = eval_results['long_tail_performance'].get('HR@5', 0)
                        long_tail_hr10 = eval_results['long_tail_performance'].get('HR@10', 0)
                        long_tail_perf = f"HR@1: {long_tail_hr1:.4f}, HR@5: {long_tail_hr5:.4f}, HR@10: {long_tail_hr10:.4f}"
                        long_tail_status = '已评估'
                    else:
                        long_tail_perf = '待评估'
                        long_tail_status = '未评估'
                    
                    # 冷启动物品性能
                    if 'cold_start_performance' in eval_results and eval_results['cold_start_performance']:
                        cold_start_hr1 = eval_results['cold_start_performance'].get('HR@1', 0)
                        cold_start_hr5 = eval_results['cold_start_performance'].get('HR@5', 0)
                        cold_start_hr10 = eval_results['cold_start_performance'].get('HR@10', 0)
                        cold_start_perf = f"HR@1: {cold_start_hr1:.4f}, HR@5: {cold_start_hr5:.4f}, HR@10: {cold_start_hr10:.4f}"
                        cold_start_status = '已评估'
                    else:
                        cold_start_perf = '待评估'
                        cold_start_status = '未评估'
                else:
                    long_tail_perf = '待评估'
                    long_tail_status = '未评估'
                    cold_start_perf = '待评估'
                    cold_start_status = '未评估'
            except Exception as e:
                long_tail_perf = '评估失败'
                long_tail_status = '错误'
                cold_start_perf = '评估失败'
                cold_start_status = '错误'
            
            f.write(f'| 长尾物品推荐 | {long_tail_perf} | {long_tail_status} |\n')
            f.write(f'| 冷启动物品推荐 | {cold_start_perf} | {cold_start_status} |\n\n')
            
            # 二、结论与建议生成
            f.write('## 二、结论与建议生成\n\n')
            
            # 1. 智能诊断
            f.write('### 1. 智能诊断\n\n')
            f.write('| 诊断项目 | 结果 |\n')
            f.write('|----------|------|\n')
            f.write(f'| 模型收敛状态 | {"已收敛" if summary["model_performance"]["converged"] else "未收敛"} |\n')
            f.write(f'| 过拟合检测 | {"是" if summary["model_performance"]["overfitting_detected"] else "否"} |\n')
            f.write(f'| 训练损失趋势 | {summary["model_performance"]["train_loss_trend"]} |\n')
            f.write(f'| 验证损失趋势 | {summary["model_performance"]["val_loss_trend"]} |\n\n')
            
            # 2. 优化建议
            f.write('### 2. 优化建议\n\n')
            for i, rec in enumerate(summary['recommendations'], 1):
                f.write(f'{i}. {rec}\n')
            f.write('\n')
            
            # 三、训练可视化
            f.write('## 三、训练可视化\n\n')
            f.write('### 1. 实时监控\n\n')
            f.write('- **TensorBoard 日志路径**: `{}`\n'.format(os.path.join(os.path.dirname(save_path), 'logs')))
            f.write('- **查看命令**: `tensorboard --logdir={}`\n'.format(os.path.join(os.path.dirname(save_path), 'logs')))
            f.write('\n')
            
            f.write('### 2. 静态复盘\n\n')
            f.write('![训练曲线图](training_curves.png)\n\n')
            
            # 四、详细训练历史
            f.write('## 四、详细训练历史\n\n')
            f.write('| Epoch | 训练损失 | 验证损失 | 验证准确率 | 学习率 |\n')
            f.write('|-------|---------|---------|-----------|--------|\n')
            for i in range(len(self.history['train_loss'])):
                train_loss = self.history['train_loss'][i]
                val_loss = self.history['val_loss'][i] if i < len(self.history['val_loss']) else 'N/A'
                val_acc = self.history['val_accuracy'][i] if i < len(self.history['val_accuracy']) else 'N/A'
                lr = self.history['learning_rate'][i] if i < len(self.history['learning_rate']) else 'N/A'
                f.write(f'| {i+1} | {train_loss:.4f} | {val_loss} | {val_acc} | {lr} |\n')
            f.write('\n')
            
            # 五、技术细节
            f.write('## 五、技术细节\n\n')
            f.write('### 1. 正则化措施\n\n')
            f.write('- L1 正则化系数: {}\n'.format(self.l1_lambda))
            f.write('- L2 正则化（权重衰减）: {}\n'.format(self.train_config.get('regularization', {}).get('l2_weight_decay', 'N/A')))
            f.write('- 早停耐心值: {}\n'.format(self.patience))
            f.write('- Dropout 率: {}\n\n'.format(self.train_config.get('model_config', {}).get('dropout', 'N/A')))
            
            f.write('### 2. 训练环境\n\n')
            f.write('- 设备: {}\n'.format(self.device))
            f.write('- PyTorch 版本: {}\n'.format(torch.__version__))
            f.write('- CUDA 可用: {}\n\n'.format(torch.cuda.is_available()))
            
            # 六、结论
            f.write('## 六、结论\n\n')
            if summary['model_performance']['overfitting_detected']:
                f.write('**⚠️ 警告:** 检测到过拟合现象，建议采取正则化措施。\n\n')
            elif not summary['model_performance']['converged']:
                f.write('**⚠️ 注意:** 模型可能未完全收敛，建议增加训练轮数或调整学习率。\n\n')
            else:
                f.write('**✅ 成功:** 训练过程正常，模型性能良好。\n\n')
            
            f.write('模型已保存到: `{}`\n'.format(save_path))
            f.write('TensorBoard 日志: `{}`\n'.format(os.path.join(os.path.dirname(save_path), 'logs')))
            f.write('训练总结文件: `{}`\n'.format(md_path))
        
        print(f"Markdown 训练总结已保存到: {md_path}")


class MetonymyPredictor:
    """预测器 - 用于训练后的预测"""
    
    def __init__(
        self,
        model_path: str,
        dataset: YelpMetonymyDataset,
        device: str = 'cuda'
    ):
        self.device = device
        self.dataset = dataset
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path: str) -> MetonymyRec:
        """加载训练好的模型"""
        # 创建模型
        model = MetonymyRec(
            item_num=len(self.dataset.item2id) + 1,
            text_dim=self.dataset.text_dim,
            image_dim=self.dataset.image_dim,
            hidden_dim=256,
            max_seq_len=self.dataset.max_seq_len,
            n_prototypes=16,
            n_heads=4,
            n_layers=2
        )
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        print(f"Model loaded from {model_path}")
        return model
    
    def predict_similarity(self, photo_id1: str, photo_id2: str) -> dict:
        """
        预测两个照片之间的相似度
        
        Args:
            photo_id1: 第一张照片ID
            photo_id2: 第二张照片ID
        
        Returns:
            相似度分数
        """
        if photo_id1 not in self.dataset.item2id or photo_id2 not in self.dataset.item2id:
            return {'error': 'Photo ID not found'}
        
        item_id1 = self.dataset.item2id[photo_id1]
        item_id2 = self.dataset.item2id[photo_id2]
        
        # 查找照片的特征
        text_feat1 = None
        image_feat1 = None
        text_feat2 = None
        image_feat2 = None
        
        for label_id, seq_data in self.dataset.label_sequences.items():
            items = seq_data['items']
            if item_id1 in items:
                idx = items.index(item_id1)
                text_feat1 = seq_data['text_features'][idx]
                image_feat1 = seq_data['image_features'][idx]
            if item_id2 in items:
                idx = items.index(item_id2)
                text_feat2 = seq_data['text_features'][idx]
                image_feat2 = seq_data['image_features'][idx]
        
        if text_feat1 is None or text_feat2 is None:
            return {'error': 'Features not found'}
        
        # 准备输入
        item_seq = torch.tensor([[item_id1]], dtype=torch.long).to(self.device)
        text_tensor = torch.tensor([[text_feat1]], dtype=torch.float32).to(self.device)
        image_tensor = torch.tensor([[image_feat1]], dtype=torch.float32).to(self.device)
        mask = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor([item_id2], dtype=torch.long).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(
                item_seq,
                text_tensor,
                image_tensor,
                target_tensor,
                None,
                mask
            )
            
            score = torch.sigmoid(outputs['logits']).item()
        
        return {
            'photo_id1': photo_id1,
            'photo_id2': photo_id2,
            'similarity_score': score
        }
    
    def recommend_for_photo(self, photo_id: str, top_k: int = 10) -> dict:
        """
        为指定照片推荐相似的照片
        
        Args:
            photo_id: 照片ID
            top_k: 返回Top-K推荐
        
        Returns:
            推荐结果
        """
        if photo_id not in self.dataset.item2id:
            return {'error': f'Photo {photo_id} not found'}
        
        item_id = self.dataset.item2id[photo_id]
        
        # 查找照片的特征
        text_feat = None
        image_feat = None
        
        for label_id, seq_data in self.dataset.label_sequences.items():
            items = seq_data['items']
            if item_id in items:
                idx = items.index(item_id)
                text_feat = seq_data['text_features'][idx]
                image_feat = seq_data['image_features'][idx]
                break
        
        if text_feat is None:
            return {'error': 'Features not found'}
        
        # 准备输入
        item_seq = torch.tensor([[item_id]], dtype=torch.long).to(self.device)
        text_tensor = torch.tensor([[text_feat]], dtype=torch.float32).to(self.device)
        image_tensor = torch.tensor([[image_feat]], dtype=torch.float32).to(self.device)
        mask = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)
        
        # 预测与所有其他照片的相似度（批量处理）
        all_scores = []
        all_item_ids = [target_id for target_id in range(1, len(self.dataset.item2id) + 1) if target_id != item_id]
        
        # 批量处理参数
        batch_size_items = 64
        
        with torch.no_grad():
            # 批量预测
            for j in range(0, len(all_item_ids), batch_size_items):
                batch_item_ids = all_item_ids[j:j+batch_size_items]
                if not batch_item_ids:
                    continue
                
                # 创建批次目标张量
                batch_targets = torch.tensor(batch_item_ids, dtype=torch.long).to(self.device)
                
                # 重复序列特征以匹配批次大小
                batch_seq = item_seq.repeat(len(batch_item_ids), 1)
                batch_text = text_tensor.repeat(len(batch_item_ids), 1, 1)
                batch_image = image_tensor.repeat(len(batch_item_ids), 1, 1)
                batch_mask = mask.repeat(len(batch_item_ids), 1)
                
                # 一次前向传播处理整个批次
                outputs = self.model(
                    batch_seq,
                    batch_text,
                    batch_image,
                    batch_targets,
                    None,
                    batch_mask
                )
                
                scores = torch.sigmoid(outputs['logits']).cpu().numpy()
                
                # 收集结果
                for item_id, score in zip(batch_item_ids, scores):
                    all_scores.append((item_id, score))
        
        # 排序并返回Top-K
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_items = []
        for item_id, score in all_scores[:top_k]:
            photo_id_rec = self.dataset.id2item.get(item_id, 'unknown')
            top_items.append({
                'photo_id': photo_id_rec,
                'item_id': item_id,
                'score': score
            })
        
        return {
            'query_photo_id': photo_id,
            'top_k': top_k,
            'recommendations': top_items
        }
    
    def recommend_by_label(self, label: str, top_k: int = 10) -> dict:
        """
        为指定类别推荐照片
        
        Args:
            label: 类别标签 (food, inside, outside, d rink, menu)
            top_k: 返回Top-K推荐
        
        Returns:
            推荐结果
        """

        if label not in self.dataset.label2id:
            return {'error': f'Label {label} not found', 'available_labels': list(self.dataset.label2id.keys())}
        
        label_id = self.dataset.label2id[label]
        
        if label_id not in self.dataset.label_sequences:
            return {'error': f'No data for label {label}'}
        
        seq_data = self.dataset.label_sequences[label_id]
        items = seq_data['items']
        
        if len(items) == 0:
            return {'error': f'No items for label {label}'}
        
        # 使用类别内的第一个物品作为查询
        query_item_id = items[0]
        
        # 查找特征
        text_feat = seq_data['text_features'][0]
        image_feat = seq_data['image_features'][0]
        
        # 准备输入
        item_seq = torch.tensor([[query_item_id]], dtype=torch.long).to(self.device)
        text_tensor = torch.tensor([[text_feat]], dtype=torch.float32).to(self.device)
        image_tensor = torch.tensor([[image_feat]], dtype=torch.float32).to(self.device)
        mask = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)
        
        # 预测与所有物品的相似度（批量处理）
        all_scores = []
        all_item_ids = [target_id for target_id in range(1, len(self.dataset.item2id) + 1) if target_id != query_item_id]
        
        # 批量处理参数
        batch_size_items = 64
        
        with torch.no_grad():
            # 批量预测
            for j in range(0, len(all_item_ids), batch_size_items):
                batch_item_ids = all_item_ids[j:j+batch_size_items]
                if not batch_item_ids:
                    continue
                
                # 创建批次目标张量
                batch_targets = torch.tensor(batch_item_ids, dtype=torch.long).to(self.device)
                
                # 重复序列特征以匹配批次大小
                batch_seq = item_seq.repeat(len(batch_item_ids), 1)
                batch_text = text_tensor.repeat(len(batch_item_ids), 1, 1)
                batch_image = image_tensor.repeat(len(batch_item_ids), 1, 1)
                batch_mask = mask.repeat(len(batch_item_ids), 1)
                
                # 一次前向传播处理整个批次
                outputs = self.model(
                    batch_seq,
                    batch_text,
                    batch_image,
                    batch_targets,
                    None,
                    batch_mask
                )
                
                scores = torch.sigmoid(outputs['logits']).cpu().numpy()
                
                # 收集结果
                for item_id, score in zip(batch_item_ids, scores):
                    all_scores.append((item_id, score))
        
        # 排序并返回Top-K
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_items = []
        for item_id, score in all_scores[:top_k]:
            photo_id = self.dataset.id2item.get(item_id, 'unknown')
            top_items.append({
                'photo_id': photo_id,
                'item_id': item_id,
                'score': score
            })
        
        return {
            'label': label,
            'query_photo_id': self.dataset.id2item.get(query_item_id, 'unknown'),
            'top_k': top_k,
            'recommendations': top_items
        }
    
    def evaluate(self, k_list=[5, 10, 20]) -> dict:
        """
        评估模型性能
        
        Args:
            k_list: 评估K值列表
        
        Returns:
            评估指标
        """


        # 初始化评估指标
        metrics = {
            'total_tests': 0,
            'hit_rate': {},
            'ndcg': {},
            'recall': {},
            'metonymy_accuracy': 0,
            'metonymy_tests': 0,
            'long_tail_performance': {},
            'cold_start_performance': {}
        }
        
        # 初始化各K值的指标
        for k in k_list:
            metrics['hit_rate'][f'HR@{k}'] = 0
            metrics['ndcg'][f'NDCG@{k}'] = 0
            metrics['recall'][f'Recall@{k}'] = 0
            metrics['long_tail_performance'][f'HR@{k}'] = 0
            metrics['cold_start_performance'][f'HR@{k}'] = 0
        
        # 构建物品频率统计
        item_freq = self._calculate_item_frequency()
        
        with torch.no_grad():
            for label_id, seq_data in tqdm(self.dataset.label_sequences.items(), desc="Evaluating"):
                items = seq_data['items']
                text_feats = seq_data['text_features']
                image_feats = seq_data['image_features']
                
                if len(items) < 2:
                    continue
                
                # 对类别内的每个物品进行评估
                for i in range(len(items) - 1):
                    query_item_id = items[i]
                    target_item_id = items[i+1]
                    
                    text_feat = text_feats[i]
                    image_feat = image_feats[i]
                    
                    # 准备输入
                    item_seq = torch.tensor([[query_item_id]], dtype=torch.long).to(self.device)
                    text_tensor = torch.tensor([[text_feat]], dtype=torch.float32).to(self.device)
                    image_tensor = torch.tensor([[image_feat]], dtype=torch.float32).to(self.device)
                    mask = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)
                    
                    # 预测与所有物品的相似度（批量处理）
                    all_scores = []
                    all_item_ids = [target_id for target_id in range(1, len(self.dataset.item2id) + 1) if target_id != query_item_id]
                    
                    # 批量处理参数
                    batch_size_items = 64
                    
                    # 批量预测
                    for j in range(0, len(all_item_ids), batch_size_items):
                        batch_item_ids = all_item_ids[j:j+batch_size_items]
                        if not batch_item_ids:
                            continue
                        
                        # 创建批次目标张量
                        batch_targets = torch.tensor(batch_item_ids, dtype=torch.long).to(self.device)
                        
                        # 重复序列特征以匹配批次大小
                        batch_seq = item_seq.repeat(len(batch_item_ids), 1)
                        batch_text = text_tensor.repeat(len(batch_item_ids), 1, 1)
                        batch_image = image_tensor.repeat(len(batch_item_ids), 1, 1)
                        batch_mask = mask.repeat(len(batch_item_ids), 1)
                        
                        # 一次前向传播处理整个批次
                        outputs = self.model(
                            batch_seq,
                            batch_text,
                            batch_image,
                            batch_targets,
                            None,
                            batch_mask
                        )
                        
                        scores = torch.sigmoid(outputs['logits']).cpu().numpy()
                        
                        # 收集结果
                        for item_id, score in zip(batch_item_ids, scores):
                            all_scores.append((item_id, score))
                    
                    # 排序
                    all_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # 计算评估指标
                    for k in k_list:
                        top_items = [item_id for item_id, _ in all_scores[:k]]
                        
                        # HR@k
                        if target_item_id in top_items:
                            metrics['hit_rate'][f'HR@{k}'] += 1
                        
                        # NDCG@k
                        if target_item_id in top_items:
                            rank = top_items.index(target_item_id) + 1
                            ndcg = 1 / np.log2(rank + 1)
                            metrics['ndcg'][f'NDCG@{k}'] += ndcg
                        
                        # Recall@k
                        if target_item_id in top_items:
                            metrics['recall'][f'Recall@{k}'] += 1
                        
                        # 长尾性能 (频率低于中位数的物品)
                        if item_freq.get(target_item_id, 0) < np.median(list(item_freq.values())):
                            if target_item_id in top_items:
                                metrics['long_tail_performance'][f'HR@{k}'] += 1
                        
                        # 冷启动性能 (频率为1的物品)
                        if item_freq.get(target_item_id, 0) == 1:
                            if target_item_id in top_items:
                                metrics['cold_start_performance'][f'HR@{k}'] += 1
                    
                    # 转喻理解能力评估
                    # 这里需要根据实际的转喻标注数据进行评估
                    # 示例：检查是否正确识别转喻类型
                    # 假设我们有转喻类型标注
                    metonymy_eval = self._evaluate_metonymy_understanding(query_item_id, target_item_id)
                    if metonymy_eval is not None:
                        if metonymy_eval['correct']:
                            metrics['metonymy_accuracy'] += 1
                        metrics['metonymy_tests'] += 1
                    
                    metrics['total_tests'] += 1
        
        # 计算最终指标
        total = metrics['total_tests']
        if total > 0:
            for k in k_list:
                metrics['hit_rate'][f'HR@{k}'] /= total
                metrics['ndcg'][f'NDCG@{k}'] /= total
                metrics['recall'][f'Recall@{k}'] /= total
                
                # 计算长尾和冷启动的比例
                long_tail_count = sum(1 for freq in item_freq.values() if freq < np.median(list(item_freq.values())))
                cold_start_count = sum(1 for freq in item_freq.values() if freq == 1)
                
                if long_tail_count > 0:
                    metrics['long_tail_performance'][f'HR@{k}'] /= long_tail_count
                if cold_start_count > 0:
                    metrics['cold_start_performance'][f'HR@{k}'] /= cold_start_count
        
        # 转喻准确率
        if metrics['metonymy_tests'] > 0:
            metrics['metonymy_accuracy'] /= metrics['metonymy_tests']
        
        return metrics
    
    def _calculate_item_frequency(self) -> dict:
        """
        计算物品出现频率
        
        Returns:
            物品频率字典
        """
        item_freq = {}
        for label_id, seq_data in self.dataset.label_sequences.items():
            for item_id in seq_data['items']:
                if item_id in item_freq:
                    item_freq[item_id] += 1
                else:
                    item_freq[item_id] = 1
        return item_freq
    
    def _evaluate_metonymy_understanding(self, query_item_id: int, target_item_id: int) -> dict:
        """
        评估转喻理解能力
        
        Args:
            query_item_id: 查询物品ID
            target_item_id: 目标物品ID
        
        Returns:
            转喻评估结果
        """
        # 查找查询物品和目标物品的转喻标注
        query_photo_id = self.dataset.id2item.get(query_item_id, '')
        target_photo_id = self.dataset.id2item.get(target_item_id, '')
        
        if not query_photo_id or not target_photo_id:
            return None
        
        # 从原始数据中查找转喻标注
        query_metonymy_type = None
        target_metonymy_type = None
        
        # 遍历原始数据查找转喻标注
        for _, row in self.dataset.raw_data.iterrows():
            if row['photo_id'] == query_photo_id:
                if 'metonymy_annotation' in row:
                    metonymy_annotation = row['metonymy_annotation']
                    if metonymy_annotation.get('contains_metonymy'):
                        query_metonymy_type = metonymy_annotation.get('metonymy_type')
            if row['photo_id'] == target_photo_id:
                if 'metonymy_annotation' in row:
                    metonymy_annotation = row['metonymy_annotation']
                    if metonymy_annotation.get('contains_metonymy'):
                        target_metonymy_type = metonymy_annotation.get('metonymy_type')
        
        # 评估转喻类型识别
        if query_metonymy_type and target_metonymy_type:
            # 检查转喻类型是否匹配
            if query_metonymy_type == target_metonymy_type:
                return {'correct': True, 'type': query_metonymy_type}
            else:
                return {'correct': False, 'type': query_metonymy_type, 'target_type': target_metonymy_type}
        
        return None
    
    def evaluate_explainability(self, sample_size=100) -> dict:
        """
        评估模型的可解释性
        
        Args:
            sample_size: 评估样本大小
        
        Returns:
            可解释性评估结果
        """
        # 这里需要实现人工评估的接口
        # 示例实现，实际应用中需要根据具体需求调整
        
        # 生成解释样本
        explanations = []
        count = 0
        
        with torch.no_grad():
            for label_id, seq_data in self.dataset.label_sequences.items():
                if count >= sample_size:
                    break
                
                items = seq_data['items']
                text_feats = seq_data['text_features']
                image_feats = seq_data['image_features']
                
                if len(items) < 2:
                    continue
                
                query_item_id = items[0]
                target_item_id = items[1]
                
                text_feat = text_feats[0]
                image_feat = image_feats[0]
                
                # 准备输入
                item_seq = torch.tensor([[query_item_id]], dtype=torch.long).to(self.device)
                text_tensor = torch.tensor([[text_feat]], dtype=torch.float32).to(self.device)
                image_tensor = torch.tensor([[image_feat]], dtype=torch.float32).to(self.device)
                mask = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)
                target_tensor = torch.tensor([target_item_id], dtype=torch.long).to(self.device)
                
                # 获取模型输出
                outputs = self.model(
                    item_seq,
                    text_tensor,
                    image_tensor,
                    target_tensor,
                    None,
                    mask
                )
                
                # 生成解释
                explanation = self._generate_explanation(
                    query_item_id,
                    target_item_id,
                    outputs
                )
                explanations.append(explanation)
                count += 1
        
        # 人工评估接口
        # 实际应用中，这里应该生成评估表格或界面供人工评估
        return {
            'sample_size': len(explanations),
            'explanations': explanations[:5],  # 只返回前5个示例
            'evaluation_method': '人工评估（1-5分）',
            'criteria': [
                '解释的清晰度',
                '转喻关系的准确性',
                '推荐理由的合理性',
                '语言表达的自然性',
                '整体满意度'
            ]
        }
    
    def _generate_explanation(self, query_item_id: int, target_item_id: int, outputs: dict) -> dict:
        """
        生成推荐解释
        
        Args:
            query_item_id: 查询物品ID
            target_item_id: 目标物品ID
            outputs: 模型输出
        
        Returns:
            解释文本
        """
        # 生成解释
        query_photo_id = self.dataset.id2item.get(query_item_id, 'unknown')
        target_photo_id = self.dataset.id2item.get(target_item_id, 'unknown')
        
        # 查找照片信息
        query_info = {}
        target_info = {}
        
        for _, row in self.dataset.raw_data.iterrows():
            if row['photo_id'] == query_photo_id:
                # 从转喻标注中获取情感信息
                sentiment = 'neutral'
                if 'metonymy_annotation' in row:
                    sentiment = row['metonymy_annotation'].get('inferred_sentiment', 'neutral')
                
                query_info = {
                    'caption': row.get('caption', ''),
                    'label': row.get('label', ''),
                    'sentiment': sentiment
                }
            if row['photo_id'] == target_photo_id:
                # 从转喻标注中获取情感信息
                sentiment = 'neutral'
                if 'metonymy_annotation' in row:
                    sentiment = row['metonymy_annotation'].get('inferred_sentiment', 'neutral')
                
                target_info = {
                    'caption': row.get('caption', ''),
                    'label': row.get('label', ''),
                    'sentiment': sentiment
                }
        
        # 生成解释文本
        explanation = f"推荐理由："
        explanation += f"基于照片 {query_photo_id} 的特征，"
        explanation += f"其描述为 '{query_info.get('caption', '无描述')}'，"
        explanation += f"类别为 {query_info.get('label', '未知')}，"
        explanation += f"情感倾向为 {query_info.get('sentiment', '中性')}。"
        explanation += f"推荐照片 {target_photo_id}，"
        explanation += f"其描述为 '{target_info.get('caption', '无描述')}'，"
        explanation += f"类别为 {target_info.get('label', '未知')}，"
        explanation += f"情感倾向为 {target_info.get('sentiment', '中性')}。"
        
        return {
            'query_photo_id': query_photo_id,
            'target_photo_id': target_photo_id,
            'explanation': explanation,
            'score': torch.sigmoid(outputs['logits']).item()
        }


def pretrain_model(model, train_loader, val_loader, args, device='cuda'):
    """
    预训练阶段：对比学习 + 情感对齐 + 多样性正则
    
    Args:
        model: MetonymyRecPretrain模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        args: 预训练参数
        device: 设备
    
    Returns:
        预训练后的模型
    """
    from src.models.metonymy_rec import MetonymyRecPretrain
    
    if not isinstance(model, MetonymyRecPretrain):
        print("错误：预训练需要使用MetonymyRecPretrain模型")
        return model
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.get('pretrain_lr', 1e-4), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.get('pretrain_epochs', 10))
    
    best_val_loss = float('inf')
    
    for epoch in range(args.get('pretrain_epochs', 10)):
        model.train()
        total_loss = 0
        total_contrastive_loss = 0
        total_sentiment_loss = 0
        total_diversity_loss = 0
        batch_count = 0
        
        for batch in tqdm(train_loader, desc=f"预训练 Epoch {epoch+1}/{args.get('pretrain_epochs', 10)}"):
            item_seq = batch['item_seq'].to(device)
            text_features = batch['text_features'].to(device)
            image_features = batch['image_features'].to(device)
            sentiment_ids = batch['sentiment_id'].to(device)
            metonymy_type_ids = batch['metonymy_type_id'].to(device)
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            # 1. 序列-物品对比学习（正样本：同一序列中相邻物品；负样本：随机采样）
            anchor_emb = model.get_contrastive_embeddings(item_seq)  # (batch_size, 128)
            
            # 正样本：同一序列的下一个物品
            batch_size, seq_len = item_seq.shape
            positive_emb = model.get_contrastive_embeddings(item_seq[:, 1:].contiguous())
            
            # 负样本：随机采样不同序列的物品
            n_neg = 4
            negative_embs = []
            for i in range(batch_size):
                # 为每个样本创建 n_neg 个负样本，保持形状为 (1, n_neg)
                neg_item = torch.randint(1, item_seq.max(), (1, n_neg)).to(device)
                neg_emb = model.get_contrastive_embeddings(neg_item)
                negative_embs.append(neg_emb)
            negative_embs = torch.stack(negative_embs, dim=0)  # (batch_size, n_neg, 128)
            
            contrastive_loss = model.compute_contrastive_loss(
                anchor_emb,
                positive_emb,
                negative_embs
            )
            
            # 2. 转喻-情感对齐（使用转喻增强字幕与情感标签对齐）
            # 获取用户表示
            target_item = item_seq[:, -1:]
            target_tensor = target_item.squeeze(-1)
            
            outputs = model(
                item_seq,
                text_features,
                image_features,
                target_tensor,
                None,
                mask
            )
            
            user_rep = outputs['user_representation']  # (batch_size, hidden_dim)
            
            # 情感分类损失
            sentiment_logits = model.predict_sentiment(user_rep)  # (batch_size, 3)
            sentiment_loss = F.cross_entropy(sentiment_logits, sentiment_ids)
            
            # 转喻类型分类损失
            metonymy_logits = model.predict_metonymy_type(user_rep)  # (batch_size, 4)
            metonymy_loss = F.cross_entropy(metonymy_logits, metonymy_type_ids)
            
            # 3. 兴趣原型多样性正则
            diversity_loss = outputs.get('diversity_loss', torch.tensor(0.0).to(device))
            
            # 总损失
            loss = (
                contrastive_loss * 0.5 +
                sentiment_loss * 0.3 +
                metonymy_loss * 0.2 +
                0.01 * diversity_loss
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_sentiment_loss += sentiment_loss.item()
            total_diversity_loss += diversity_loss.item()
            batch_count += 1
        
        scheduler.step()
        
        avg_loss = total_loss / batch_count
        avg_contrastive = total_contrastive_loss / batch_count
        avg_sentiment = total_sentiment_loss / batch_count
        avg_diversity = total_diversity_loss / batch_count
        
        print(f"预训练 Epoch {epoch+1}: Loss={avg_loss:.4f}, Contrastive={avg_contrastive:.4f}, "
              f"Sentiment={avg_sentiment:.4f}, Diversity={avg_diversity:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")
        
        # 验证
        model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                item_seq = batch['item_seq'].to(device)
                text_features = batch['text_features'].to(device)
                image_features = batch['image_features'].to(device)
                sentiment_ids = batch['sentiment_id'].to(device)
                mask = batch['mask'].to(device)
                
                target_item = item_seq[:, -1:]
                target_tensor = target_item.squeeze(-1)
                
                outputs = model(
                    item_seq,
                    text_features,
                    image_features,
                    target_tensor,
                    None,
                    mask
                )
                
                sentiment_logits = model.predict_sentiment(outputs['user_representation'])
                sentiment_loss = F.cross_entropy(sentiment_logits, sentiment_ids)
                val_loss += sentiment_loss.item()
                val_count += 1
        
        avg_val_loss = val_loss / max(val_count, 1)
        print(f"  验证损失: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            pretrain_path = os.path.join(args.get('save_path', './checkpoints'), 'pretrain_best.pt')
            torch.save(model.state_dict(), pretrain_path)
            print(f"  最佳预训练模型已保存")
    
    # 加载最佳预训练模型
    pretrain_path = os.path.join(args.get('save_path', './checkpoints'), 'pretrain_best.pt')
    if os.path.exists(pretrain_path):
        model.load_state_dict(torch.load(pretrain_path, map_location=device))
        print(f"已加载最佳预训练模型")
    
    return model


def train_model(data_path, save_path, epochs=50, batch_size=8, lr=1e-3, device=None,
                pretrain_epochs=0, freeze_backbone=False):
    """
    训练MetonymyRec模型
    
    Args:
        data_path: 训练数据路径 (JSON Lines格式)
        save_path: 模型保存路径
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        device: 训练设备 (cuda/cpu)
    """
    # 自动选择设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误：数据文件不存在: {data_path}")
        return
    
    print(f"加载训练数据: {data_path}")
    
    # 创建训练数据集（支持数据划分）
    dataset = YelpMetonymyDataset(
        data_path=data_path,
        max_seq_len=5,
        text_dim=768,
        image_dim=1024,
        split='train',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # 创建验证集和测试集
    val_dataset = YelpMetonymyDataset(
        data_path=data_path,
        max_seq_len=5,
        text_dim=768,
        image_dim=1024,
        split='val',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    test_dataset = YelpMetonymyDataset(
        data_path=data_path,
        max_seq_len=5,
        text_dim=768,
        image_dim=1024,
        split='test',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    print(f"数据集大小: 训练集 {len(dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")
    print(f"类别数量: {len(dataset.label_sequences)}")
    print(f"物品数量: {len(dataset.item2id)}")
    
    if len(dataset) == 0:
        print("错误：数据集为空")
        return
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )
    
    print(f"训练样本数: {len(dataset)}, 验证样本数: {len(val_dataset)}")
    
    # 创建模型
    use_pretrain = pretrain_epochs > 0
    
    if use_pretrain:
        print("创建MetonymyRecPretrain模型（支持预训练）...")
        from src.models.metonymy_rec import MetonymyRecPretrain
        model = MetonymyRecPretrain(
            item_num=len(dataset.item2id) + 1,
            text_dim=768,
            image_dim=1024,
            hidden_dim=256,
            max_seq_len=5,
            n_prototypes=16,
            n_heads=4,
            n_layers=2,
            n_sentiment_classes=3
        )
    else:
        print("创建MetonymyRec模型...")
        model = MetonymyRec(
            item_num=len(dataset.item2id) + 1,
            text_dim=768,
            image_dim=1024,
            hidden_dim=256,
            max_seq_len=5,
            n_prototypes=16,
            n_heads=4,
            n_layers=2
        )
    
    # 将模型移动到指定设备
    model = model.to(device)
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 预训练阶段
    if use_pretrain:
        print(f"\n{'='*50}")
        print(f"开始预训练阶段，共 {pretrain_epochs} 轮...")
        print(f"{'='*50}")
        
        pretrain_args = {
            'pretrain_epochs': pretrain_epochs,
            'pretrain_lr': lr,
            'save_path': save_path
        }
        
        model = pretrain_model(model, train_loader, val_loader, pretrain_args, device)
        
        print(f"\n预训练完成！")
        
        # 如果需要冻结主体参数，仅微调适配器
        if freeze_backbone:
            print("冻结主体参数，仅微调适配器...")
            # 冻结非适配器参数（保留contrastive_head、sentiment_classifier、metonymy_type_classifier可训练）
            trainable_prefixes = ['contrastive_head', 'sentiment_classifier', 'metonymy_type_classifier', 'temperature']
            for name, param in model.named_parameters():
                is_trainable = any(prefix in name for prefix in trainable_prefixes)
                param.requires_grad = is_trainable

            # 统计可训练参数
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"可训练参数数量: {trainable_params:,}")
            if trainable_params == 0:
                print("警告：没有可训练参数，请检查freeze_backbone设置")
    
    # 准备训练配置
    train_config = {
        'data_path': data_path,
        'save_path': save_path,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'device': device,
        'regularization': {
            'l1_lambda': 0.0001,  # L1正则化系数
            'l2_weight_decay': 1e-5,  # L2正则化（权重衰减）
            'early_stopping': {
                'patience': 5,  # 早停耐心值
                'enabled': True  # 启用早停
            }
        },
        'model_config': {
            'item_num': len(dataset.item2id) + 1,
            'text_dim': 768,
            'image_dim': 1024,
            'hidden_dim': 256,
            'max_seq_len': 5,
            'n_prototypes': 16,
            'n_heads': 4,
            'n_layers': 2,
            'dropout': 0.1  # 模型dropout率
        },
        'dataset_info': {
            'total_samples': len(dataset) + len(val_dataset) + len(test_dataset),
            'train_size': len(dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'num_classes': len(dataset.label_sequences),
            'num_items': len(dataset.item2id)
        }
    }
    
    # 创建训练器
    trainer = MetonymyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset=dataset,
        lr=lr,
        weight_decay=train_config['regularization']['l2_weight_decay'],
        device=device,
        train_config=train_config,
        patience=train_config['regularization']['early_stopping']['patience'],
        l1_lambda=train_config['regularization']['l1_lambda']
    )
    
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 开始训练
    print(f"\n开始训练，共 {epochs} 轮...")
    print("=" * 50)
    
    try:
        trainer.train(epochs=epochs, save_path=save_path)
        print("\n训练完成！")
        print(f"模型已保存到: {save_path}")
        
        # 保存最终模型
        final_path = os.path.join(save_path, "final_model.pt")
        torch.save(model.state_dict(), final_path)
        print(f"最终模型已保存到: {final_path}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        interrupt_path = os.path.join(save_path, "interrupted_model.pt")
        torch.save(model.state_dict(), interrupt_path)
        print(f"中断时的模型已保存到: {interrupt_path}")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def predict(model_path, data_path, mode='photo', query=None, top_k=10, device=None):
    """
    使用训练好的模型进行预测
    
    Args:
        model_path: 模型路径
        data_path: 数据路径
        mode: 预测模式 ('photo', 'label', 'similarity', 'evaluate', 'explainability', 'all')
        query: 查询参数（photo_id, label, 或 photo_id1,photo_id2）
        top_k: Top-K推荐
        device: 设备
    """
    # 自动选择设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载数据集
    print(f"加载数据: {data_path}")
    dataset = YelpMetonymyDataset(
        data_path=data_path,
        max_seq_len=5,
        text_dim=768,
        image_dim=1024
    )
    
    # 创建预测器
    predictor = MetonymyPredictor(
        model_path=model_path,
        dataset=dataset,
        device=device
    )
    
    if mode == 'photo':
        # 为指定照片推荐相似照片
        if not query:
            # 如果没有指定，使用第一个照片
            query = dataset.id2item.get(1, '')
        
        result = predictor.recommend_for_photo(query, top_k=top_k)
        print(f"\n为照片 {query} 的推荐结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    
    elif mode == 'label':
        # 为指定类别推荐照片
        if not query:
            query = 'food'  # 默认类别
        
        result = predictor.recommend_by_label(query, top_k=top_k)
        print(f"\n为类别 {query} 的推荐结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    
    elif mode == 'similarity':
        # 计算两个照片的相似度
        if not query or ',' not in query:
            print("错误：similarity模式需要提供两个photo_id，用逗号分隔")
            return
        
        photo_id1, photo_id2 = query.split(',')
        result = predictor.predict_similarity(photo_id1.strip(), photo_id2.strip())
        print(f"\n照片相似度:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    
    elif mode == 'explainability':
        # 评估模型可解释性
        print("\n评估模型可解释性...")
        result = predictor.evaluate_explainability()
        print("\n可解释性评估结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    
    elif mode == 'evaluate':
        # 评估模型（使用测试集）
        print("\n评估模型性能（使用测试集）...")
        metrics = predictor.evaluate()
        print("\n评估结果:")
        print(json.dumps(metrics, indent=2))
        return metrics
    
    elif mode == 'evaluate_val':
        # 使用验证集评估
        print("\n评估模型性能（使用验证集）...")
        val_dataset = YelpMetonymyDataset(
            data_path=data_path,
            max_seq_len=5,
            text_dim=768,
            image_dim=1024,
            split='val',
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        val_predictor = MetonymyPredictor(
            model_path=model_path,
            dataset=val_dataset,
            device=device
        )
        metrics = val_predictor.evaluate()
        print("\n验证集评估结果:")
        print(json.dumps(metrics, indent=2))
        return metrics
    
    elif mode == 'all':
        # 预测所有结果并生成总结
        print("\n开始预测所有结果...")
        all_results = {}
        
        try:
            # 1. 评估模型性能
            print("\n1. 评估模型性能...")
            all_results['evaluation'] = predictor.evaluate()
        except Exception as e:
            print(f"评估模型性能时出错: {e}")
            all_results['evaluation'] = {"error": str(e)}
        
        try:
            # 2. 评估模型可解释性
            print("\n2. 评估模型可解释性...")
            all_results['explainability'] = predictor.evaluate_explainability()
        except Exception as e:
            print(f"评估模型可解释性时出错: {e}")
            all_results['explainability'] = {"error": str(e)}
        
        try:
            # 3. 为前5个照片推荐相似照片
            print("\n3. 为前5个照片推荐相似照片...")
            all_results['photo_recommendations'] = {}
            for i in range(1, min(6, len(dataset.item2id) + 1)):
                photo_id = dataset.id2item.get(i, '')
                if photo_id:
                    try:
                        result = predictor.recommend_for_photo(photo_id, top_k=top_k)
                        # 添加图片路径信息
                        if 'recommendations' in result:
                            for item in result['recommendations']:
                                # 假设照片存储在 ./data/Yelp/Yelp_photos~/ 目录下
                                item['image_path'] = f"./data/Yelp/Yelp_photos~/{item.get('photo_id', '')}.jpg"
                        all_results['photo_recommendations'][photo_id] = result
                    except Exception as e:
                        print(f"为照片 {photo_id} 推荐时出错: {e}")
                        all_results['photo_recommendations'][photo_id] = {"error": str(e)}
        except Exception as e:
            print(f"生成照片推荐时出错: {e}")
            all_results['photo_recommendations'] = {"error": str(e)}
        
        try:
            # 4. 为每个类别推荐照片
            print("\n4. 为每个类别推荐照片...")
            all_results['label_recommendations'] = {}
            for label_id, label_data in dataset.label_sequences.items():
                label_name = label_data['label_name']
                try:
                    result = predictor.recommend_by_label(label_name, top_k=top_k)
                    # 添加图片路径信息
                    if 'recommendations' in result:
                        for item in result['recommendations']:
                            item['image_path'] = f"./data/Yelp/Yelp_photos~/{item.get('photo_id', '')}.jpg"
                    all_results['label_recommendations'][label_name] = result
                except Exception as e:
                    print(f"为类别 {label_name} 推荐时出错: {e}")
                    all_results['label_recommendations'][label_name] = {"error": str(e)}
        except Exception as e:
            print(f"生成类别推荐时出错: {e}")
            all_results['label_recommendations'] = {"error": str(e)}
        
        try:
            # 5. 计算前3对照片的相似度
            print("\n5. 计算前3对照片的相似度...")
            all_results['similarity_calculations'] = []
            photo_ids = list(dataset.item2id.keys())[:6]
            for i in range(0, len(photo_ids), 2):
                if i + 1 < len(photo_ids):
                    photo_id1 = photo_ids[i]
                    photo_id2 = photo_ids[i + 1]
                    try:
                        result = predictor.predict_similarity(photo_id1, photo_id2)
                        # 添加图片路径信息
                        result['image_path1'] = f"./data/Yelp/Yelp_photos~/{photo_id1}.jpg"
                        result['image_path2'] = f"./data/Yelp/Yelp_photos~/{photo_id2}.jpg"
                        all_results['similarity_calculations'].append(result)
                    except Exception as e:
                        print(f"计算照片 {photo_id1} 和 {photo_id2} 相似度时出错: {e}")
                        all_results['similarity_calculations'].append({"error": str(e)})
        except Exception as e:
            print(f"计算相似度时出错: {e}")
            all_results['similarity_calculations'] = [{"error": str(e)}]
        
        # 保存所有结果到JSON文件
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        # 确保输出目录为output文件夹（使用绝对路径）
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f'prediction_results_{timestamp}.json')

        try:

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            print(f"\n所有预测结果已保存到: {json_path}")

        except Exception as e:
            print(f"保存结果到JSON文件时出错: {e}")
        
        # 生成Markdown总结文件
        md_path = os.path.join(output_dir, f'prediction_summary_{timestamp}.md')
        
        try:
            generate_prediction_summary(all_results, md_path, model_path, data_path)
            print(f"预测总结已保存到: {md_path}")
        except Exception as e:
            print(f"生成Markdown总结时出错: {e}")
        
        return all_results
    
    else:
        # 默认为评估模型
        print("\n评估模型性能...")
        metrics = predictor.evaluate()
        print("\n评估结果:")
        print(json.dumps(metrics, indent=2))
        return metrics


def generate_prediction_summary(results, output_path, model_path, data_path):
    """
    生成预测结果的Markdown总结
    
    Args:
        results: 预测结果
        output_path: 输出文件路径
        model_path: 模型路径
        data_path: 数据路径
    """
    import datetime
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('# MetonymyRec 模型预测总结\n\n')
        f.write(f'**生成时间**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'**模型路径**: {model_path}\n')
        f.write(f'**数据路径**: {data_path}\n\n')
        
        # 1. 模型评估结果
        f.write('## 1. 模型评估结果\n\n')
        if 'evaluation' in results:
            eval_result = results['evaluation']
            f.write('| 指标 | 值 |\n')
            f.write('|------|-----|\n')
            for key, value in eval_result.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        f.write(f'| {key}.{sub_key} | {sub_value} |\n')
                else:
                    f.write(f'| {key} | {value} |\n')
        f.write('\n')
        
        # 2. 可解释性评估
        f.write('## 2. 可解释性评估\n\n')
        if 'explainability' in results:
            expl_result = results['explainability']
            f.write('| 指标 | 值 |\n')
            f.write('|------|-----|\n')
            for key, value in expl_result.items():
                f.write(f'| {key} | {value} |\n')
        f.write('\n')
        
        # 3. 照片推荐结果
        f.write('## 3. 照片推荐结果\n\n')
        if 'photo_recommendations' in results:
            for photo_id, recommendation in results['photo_recommendations'].items():
                f.write(f'### 为照片 {photo_id} 的推荐\n\n')
                f.write('| 排名 | 照片ID | 相似度 | 图片路径 |\n')
                f.write('|------|--------|--------|----------|\n')
                if 'recommendations' in recommendation:
                    for i, item in enumerate(recommendation['recommendations'], 1):
                        f.write(f'| {i} | {item.get("photo_id", "-")} | {item.get("similarity_score", 0):.4f} | {item.get("image_path", "-")} |\n')
                f.write('\n')
        
        # 4. 类别推荐结果
        f.write('## 4. 类别推荐结果\n\n')
        if 'label_recommendations' in results:
            for label, recommendation in results['label_recommendations'].items():
                f.write(f'### 为类别 {label} 的推荐\n\n')
                f.write('| 排名 | 照片ID | 相似度 | 图片路径 |\n')
                f.write('|------|--------|--------|----------|\n')
                if 'recommendations' in recommendation:
                    for i, item in enumerate(recommendation['recommendations'], 1):
                        f.write(f'| {i} | {item.get("photo_id", "-")} | {item.get("similarity_score", 0):.4f} | {item.get("image_path", "-")} |\n')
                f.write('\n')
        
        # 5. 相似度计算结果
        f.write('## 5. 相似度计算结果\n\n')
        if 'similarity_calculations' in results:
            f.write('| 照片1 | 照片2 | 相似度 | 图片路径1 | 图片路径2 |\n')
            f.write('|-------|-------|--------|-----------|-----------|\n')
            for result in results['similarity_calculations']:
                f.write(f'| {result.get("photo_id1", "-")} | {result.get("photo_id2", "-")} | {result.get("similarity_score", 0):.4f} | {result.get("image_path1", "-")} | {result.get("image_path2", "-")} |\n')
        f.write('\n')
        
        # 6. 总结与建议
        f.write('## 6. 总结与建议\n\n')
        f.write('### 模型性能分析\n\n')
        if 'evaluation' in results:
            eval_result = results['evaluation']
            accuracy = eval_result.get('accuracy', 0)
            if accuracy >= 0.8:
                f.write('✅ 模型性能良好，准确率较高\n')
            elif accuracy >= 0.6:
                f.write('⚠️ 模型性能一般，建议进一步优化\n')
            else:
                f.write('❌ 模型性能较差，需要大幅改进\n')
        
        f.write('\n### 可解释性分析\n\n')
        if 'explainability' in results:
            expl_result = results['explainability']
            avg_score = expl_result.get('average_score', 0)
            if avg_score >= 4.0:
                f.write('✅ 模型可解释性强，生成的解释质量高\n')
            elif avg_score >= 3.0:
                f.write('⚠️ 模型可解释性一般，生成的解释质量中等\n')
            else:
                f.write('❌ 模型可解释性差，生成的解释质量低\n')
        
        f.write('\n### 优化建议\n\n')
        f.write('1. 增加训练数据量，特别是转喻标注数据\n')
        f.write('2. 调整模型超参数，如学习率、批次大小等\n')
        f.write('3. 尝试不同的转喻特征融合方式\n')
        f.write('4. 增加模型复杂度，如添加更多层或注意力机制\n')
        f.write('5. 引入对比学习，增强模型的判别能力\n')


def main():
    
    parser = argparse.ArgumentParser(description="训练和使用MetonymyRec模型")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument(
        "--data-path",
        type=str,
        default="./data/Yelp/yelp_photos~/photos_with_metonymy_annotations.json",
        help="训练数据文件路径 (JSON Lines格式)"
    )
    train_parser.add_argument(
        "--save-path",
        type=str,
        default="./checkpoints",
        help="模型保存路径"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练轮数 (默认: 50)"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批次大小 (默认: 8)"
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率 (默认: 1e-3)"
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="训练设备 (cuda/cpu，默认自动选择)"
    )
    train_parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=0,
        help="预训练轮数 (默认: 0，不使用预训练)"
    )
    train_parser.add_argument(
        "--freeze-backbone",
        type=bool,
        default=False,
        help="冻结主体参数，仅微调适配器 (默认: False)"
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="权重衰减（L2正则化） (默认: 1e-5)"
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="早停耐心值 (默认: 5)"
    )
    train_parser.add_argument(
        "--l1-lambda",
        type=float,
        default=0.0001,
        help="L1正则化系数 (默认: 0.0001)"
    )
    
    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='使用模型进行预测')
    predict_parser.add_argument(
        "--model-path",
        type=str,
        default="./checkpoints/best_model.pt",
        required=True,
        help="训练好的模型路径"
    )
    predict_parser.add_argument(
        "--data-path",
        type=str,
        default="./data/Yelp/yelp_photos~/photos_with_metonymy_annotations.json",
        help="数据文件路径"
    )
    predict_parser.add_argument(
        "--mode",
        type=str,
        default='photo',
        choices=['all','photo', 'label', 'similarity', 'evaluate', 'evaluate_val', 'explainability'],
        help="预测模式 (photo: 为照片推荐, label: 为类别推荐, similarity: 计算相似度, evaluate: 测试集评估, evaluate_val: 验证集评估, explainability: 评估可解释性)"
    )
    predict_parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="查询参数 (photo_id, label, 或 photo_id1,photo_id2)"
    )
    predict_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K推荐数量 (默认: 10)"
    )
    predict_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备 (cuda/cpu，默认自动选择)"
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # 打印配置信息
        print("=" * 50)
        print("MetonymyRec 模型训练")
        print("=" * 50)
        print(f"数据路径: {args.data_path}")
        print(f"保存路径: {args.save_path}")
        print(f"训练轮数: {args.epochs}")
        print(f"批次大小: {args.batch_size}")
        print(f"学习率: {args.lr}")
        print("=" * 50)
        
        # 开始训练
        train_model(
            data_path=args.data_path,
            save_path=args.save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            pretrain_epochs=args.pretrain_epochs,
            freeze_backbone=args.freeze_backbone
        )
    
    elif args.command == 'predict':
        # 打印配置信息
        print("=" * 50)
        print("MetonymyRec 模型预测")
        print("=" * 50)
        print(f"模型路径: {args.model_path}")
        print(f"数据路径: {args.data_path}")
        print(f"模式: {args.mode}")
        print(f"查询: {args.query or '默认'}")
        print(f"Top-K: {args.top_k}")
        print("=" * 50)
        
        # 开始预测
        predict(
            model_path=args.model_path,
            data_path=args.data_path,
            mode=args.mode,
            query=args.query,
            top_k=args.top_k,
            device=args.device
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        print("请检查命令格式是否正确，例如:")
        print("  训练: python train_metonymy_rec.py train --data-path <data_path> --save-path <save_path>")
        print("  预测: python train_metonymy_rec.py predict --model-path <model_path> --mode <mode>")
