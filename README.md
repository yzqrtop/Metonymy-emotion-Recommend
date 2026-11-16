# MetonymyRec：转喻感知推荐系统

## 🚀 项目简介

MetonymyRec是一个基于转喻（Metonymy）的多模态推荐系统，通过识别和利用用户在评论中使用的情感转喻，提升推荐准确性。

### 核心特性

- **转喻自动标注**：基于Qwen3-VL的转喻识别与标注
- **转喻感知字幕生成**：基于Qwen2-VL生成包含转喻线索的字幕
- **转喻兴趣发现**：使用DPC-KNN算法挖掘用户的转喻兴趣原型
- **SPDR动态路由**：语义代理动态路由机制融合多模态特征
- **预训练+微调两阶段训练**：对比学习 + 情感对齐 + 多样性正则
- **严格数据划分**：训练/验证/测试集严格分离，避免数据泄露
- **转喻理解评估**：完整的转喻类型分类准确率评估
- **长尾/冷启动评估**：按物品频率分组的推荐效果评估

## 📁 项目结构

```
MetonymyRec_master_yzq/
├── src/                      # 源代码目录
│   ├── models/               # 模型相关文件
│   │   ├── metonymy_rec.py      # 完整的MetonymyRec模型
│   │   ├── spdr_layer.py        # SPDR融合模块
│   │   └── metonymy_interest.py # 转喻兴趣发现
│   ├── data/                 # 数据相关文件
│   │   ├── data_preparation.py  # 数据准备脚本
│   │   └── api_utils.py         # API密钥管理
│   └── metonymy/             # 转喻相关文件
│       ├── metonymy_annotator.py    # 转喻标注器
│       ├── metonymy_captioner.py    # 转喻感知字幕生成器
│       └── annotation_quality.py    # 标注质量控制
├── data/                     # 数据目录
│   └── Yelp/                 # Yelp数据集
│       └── yelp_photos~       # 照片数据
├── docs/                     # 文档目录
├── train_metonymy_rec.py     # 训练和预测脚本
├── annotate_metonymy.py      # 转喻标注脚本
├── generate_metonymy_captions.py # 转喻感知字幕生成脚本
├── README.md                 # 项目说明
└── requirements.txt          # 依赖包
```

## 🔧 安装依赖

### 基础依赖
```bash
pip install torch torchvision transformers accelerate
pip install pandas numpy scikit-learn tqdm
pip install Pillow requests
```

### 多模态模型
```bash
pip install qwen-vl-utils  # Qwen-VL相关
```

### 可视化与日志
```bash
pip install wandb tensorboard
```

### 可选：多模态特征预计算
```bash
pip install open-clip-torch
```

## 🎯 模块说明

### 1. 转喻标注器（MetonymyAnnotator）

- **功能**：使用Qwen3-VL对图文评论进行转喻标注
- **输出**：转喻类型、情感倾向、关键物体等结构化信息
- **质量控制**：支持计算标注者一致性（Cohen's Kappa）

### 2. 转喻感知字幕生成器（MetonymyAwareCaptioner）

- **功能**：基于Qwen2-VL生成包含转喻线索的字幕
- **用途**：为图片生成更丰富的情感信息，增强推荐模型的理解能力

### 3. 转喻兴趣发现（MetonymyInterestDiscovery）

- **核心算法**：DPC-KNN（基于密度峰值的KNN聚类）
- **功能**：挖掘用户在转喻空间中的兴趣原型
- **优势**：通过多样性正则项鼓励兴趣原型的差异化

### 4. SPDR融合模块（SPDRLayer）

- **功能**：使用语义代理动态路由机制融合多模态特征
- **特点**：自适应调整不同模态的权重，捕获转喻语义

### 5. MetonymyRec模型

- **架构**：整合上述模块的端到端推荐模型
- **输入**：物品ID序列、文本特征、图像特征、转喻字幕特征
- **输出**：用户对目标物品的偏好得分

## 📊 训练流程

1. **数据准备**：
   - 构建包含用户评论、图片、评分的数据集
   - 生成转喻标注和转喻感知字幕
   - 严格划分训练/验证/测试集（8:1:1）

2. **预训练阶段**：
   - 初始化MetonymyRecPretrain模型
   - 执行对比学习（序列-物品对比）
   - 转喻-情感对齐（情感分类）
   - 转喻类型分类
   - 兴趣原型多样性正则

3. **微调阶段**：
   - 冻结预训练主体参数
   - 仅微调多模态适配器
   - 主任务：推荐预测

4. **模型评估**：
   - 推荐效果评估：HR@k、NDCG@k、Recall@k
   - 转喻理解评估：转喻类型分类准确率
   - 长尾/冷启动评估：低频物品推荐效果
   - 基于验证集性能保存最佳模型

## 🤖 使用示例

### 1. 转喻标注

```python
from src.metonymy.metonymy_annotator import MetonymyAnnotator

annotator = MetonymyAnnotator(api_key="your_api_key")
annotation = annotator.annotate(
    review_text="这杯咖啡绝了！",
    image_path="path/to/image.jpg",
    rating=5.0
)
print(annotation)
```

### 2. 生成转喻感知字幕

```python
from src.metonymy.metonymy_captioner import MetonymyAwareCaptioner
from PIL import Image

captioner = MetonymyAwareCaptioner()
image = Image.open("path/to/image.jpg")
caption = captioner.generate_caption(image, review_text="周末的brunch时光")
print(caption)
```

### 3. 训练模型

#### 3.1 使用命令行训练

```bash
# 基本训练命令（端到端）
python train_metonymy_rec.py --data_path ./data/Yelp/yelp_photos~/photos_with_metonymy_annotations.json --epochs 10 --batch_size 16 --save_path ./output

# 预训练+微调两阶段训练
python train_metonymy_rec.py \
    --data_path ./data/Yelp/yelp_photos~/photos_with_metonymy_annotations.json \
    --pretrain_epochs 10 \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --patience 5 \
    --l1_lambda 0.0001 \
    --freeze_backbone True \
    --save_path ./output
```

#### 3.2 命令行参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_path` | 训练数据路径 | 必须指定 |
| `--epochs` | 微调训练轮数 | 10 |
| `--pretrain_epochs` | 预训练轮数 | 0（不使用预训练） |
| `--batch_size` | 批次大小 | 16 |
| `--lr` | 学习率 | 1e-4 |
| `--weight_decay` | 权重衰减（L2正则化） | 1e-5 |
| `--patience` | 早停耐心值 | 5 |
| `--l1_lambda` | L1正则化系数 | 0.0001 |
| `--freeze_backbone` | 冻结主体参数，仅微调适配器 | False |
| `--save_path` | 模型保存路径 | ./output |
| `--device` | 训练设备 (cuda/cpu) | 自动检测 |

#### 3.3 训练过程可视化

训练过程中，系统会自动生成以下可视化内容：

- **TensorBoard 实时监控**：记录批次损失、学习率、模型参数的梯度与分布
  ```bash
  tensorboard --logdir=./output/logs
  ```

- **静态训练曲线图**：训练结束后生成 `training_curves.png`，包含损失曲线和准确率曲线

- **训练总结文件**：生成 `training_summary.md` 和 `training_summary.json`，包含训练信息、性能指标和优化建议

### 4. 模型预测

#### 4.1 预测相似度

```python
from train_metonymy_rec import YelpMetonymyDataset, MetonymyPredictor

# 加载数据集
dataset = YelpMetonymyDataset(
    data_path="./data/Yelp/yelp_photos~/photos_with_metonymy_annotations.json"
)

# 创建预测器
predictor = MetonymyPredictor(
    model_path="./output/best_model.pt",
    dataset=dataset,
    device="cuda"
)

# 预测两个照片之间的相似度
result = predictor.predict_similarity(
    photo_id1="photo1_id",
    photo_id2="photo2_id"
)
print(f"相似度分数: {result['similarity_score']}")
```

#### 4.2 推荐相似照片

```python
# 为指定照片推荐相似的照片
recommendations = predictor.recommend_for_photo(
    photo_id="photo_id",
    top_k=10
)

print("推荐结果:")
for i, item in enumerate(recommendations['recommendations'], 1):
    print(f"{i}. 照片ID: {item['photo_id']}, 相似度: {item['similarity_score']:.4f}")
```

#### 4.3 预测所有结果并生成总结

使用 `all` 模式可以预测所有结果并生成详细的 Markdown 总结文件：

```bash
# 预测所有结果并生成总结
python train_metonymy_rec.py predict \
    --model-path ./output/best_model.pt \
    --mode all \
    --top-k 10
```

#### 4.4 验证集评估

使用 `evaluate_val` 模式在验证集上评估模型性能：

```bash
# 在验证集上评估模型
python train_metonymy_rec.py predict \
    --model-path ./output/best_model.pt \
    --mode evaluate_val
```

#### 4.5 测试集评估

使用 `evaluate` 模式在测试集上评估模型性能：

```bash
# 在测试集上评估模型
python train_metonymy_rec.py predict \
    --model-path ./output/best_model.pt \
    --mode evaluate
```

**功能说明**：
- 自动评估模型性能（HR@k、NDCG@k、Recall@k）
- 评估转喻理解能力（转喻类型分类准确率）
- 评估长尾和冷启动性能（低频物品推荐效果）
- 评估模型可解释性
- 为前5个照片推荐相似照片
- 为每个类别推荐照片
- 计算前3对照片的相似度
- 保存所有结果到 JSON 文件
- 生成详细的 Markdown 总结文件

**输出文件**：
- `prediction_results_<timestamp>.json`：所有预测结果
- `prediction_summary_<timestamp>.md`：详细的 Markdown 总结

## 📝 注意事项

1. **API密钥**：使用Qwen3-VL和Qwen2-VL需要配置API密钥
2. **硬件要求**：训练模型需要GPU支持，推荐至少8GB显存
3. **数据格式**：数据集需要包含photo_id、business_id、caption、label、metonymy_annotation等字段
4. **预训练特征**：需要预先计算文本和图像的特征表示
5. **训练可视化**：训练过程中会自动生成TensorBoard日志，可通过 `tensorboard --logdir=./output/logs` 查看
6. **模型保存**：训练过程中会保存最佳模型和检查点，默认保存在 `--save_path` 指定的目录
7. **预测数据**：使用 `MetonymyPredictor` 进行预测时，需要使用与训练相同的数据集对象
8. **图片路径**：转喻标注和字幕生成时，需要确保图片路径正确，建议使用绝对路径

## 📦 数据准备


### 1. 下载数据集

使用提供的数据准备脚本下载Amazon Reviews 2023或Yelp数据集：

```bash
# 下载Amazon Reviews 2023 (Electronics类别)
python src/data/data_preparation.py --dataset amazon --category Electronics

# 下载Yelp数据集
python src/data/data_preparation.py --dataset yelp
```

### 2. 配置API密钥

使用以下命令配置Qwen-VL的API密钥：

```bash
python src/data/data_preparation.py --api-key YOUR_API_KEY
```

### 3. 数据格式说明

处理后的数据格式包含以下字段：
- `user_id`：用户ID
- `item_id`：物品ID
- `timestamp`：时间戳
- `rating`：评分
- `review_text`：评论文本
- `text_feature`：文本特征（768维）
- `image_feature`：图像特征（1024维）

### 4. 特征预计算

需要使用预训练模型计算文本和图像的特征表示。以下是一个简单的示例：

```python
# 文本特征提取（使用BERT）
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def extract_text_feature(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

# 图像特征提取（使用CLIP）
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

def extract_image_feature(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs.squeeze().detach().numpy()
```

## 📚 参考文献

1. [MISSRec: Multi-Interest Self-Supervised Recommendation with Social Information](https://arxiv.org/abs/2305.01675)
2. [MUSE: Multi-modal User Intent Understanding with Self-supervised Learning](https://arxiv.org/abs/2309.07597)

## 🔄 版本更新

- **v1.2**：添加预训练+微调两阶段训练和完整评估体系
  - 实现 MetonymyRecPretrain 预训练模型
  - 支持对比学习、情感对齐和转喻类型分类
  - 严格数据划分（训练/验证/测试集）
  - 完善转喻理解评估（转喻类型分类准确率）
  - 增强长尾/冷启动评估（低频物品推荐效果）
  - 基于验证集性能保存最佳模型
  - 更新 README.md 文档

- **v1.1**：添加训练过程可视化和预测功能
  - 集成 TensorBoard 实时监控
  - 生成训练曲线图和总结文件
  - 实现 MetonymyPredictor 预测器
  - 更新 README.md 文档

- **v1.0**：初始版本，实现完整的转喻感知推荐系统

---

**作者**：MetonymyRec Team
**日期**：2026-04-19