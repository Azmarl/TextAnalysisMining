# 医学文本挖掘系统

## 项目概述

本项目针对医学文本进行挖掘和分析，包含两个主要任务和一个交互式Web应用：

1. **医学实体抽取**：从非结构化医学文本中提取医学实体，如疾病、症状、药物等，共9大类实体。
2. **临床筛选标准短文本分类**：对临床试验筛选标准进行分类，共44种语义类别。
3. **交互式Web应用**：提供用户友好的界面，支持实时实体抽取和文本分类。

## 目录结构

```
├── src/                  # 源代码目录
│   ├── data/             # 数据处理模块
│   │   └── data_loader.py # 数据加载和预处理
│   ├── models/           # 模型模块
│   │   ├── entity_extraction.py # 医学实体抽取模型
│   │   └── text_classification.py # 文本分类模型
│   ├── evaluation/       # 评估模块
│   │   └── evaluator.py  # 评估指标计算和结果保存
│   ├── utils/            # 工具模块
│   ├── config/           # 配置模块
│   └── main.py           # 主程序入口
├── CMeEE-V2/             # 医学实体抽取数据集
│   ├── CMeEE-V2_train.json # 训练集
│   ├── CMeEE-V2_dev.json   # 验证集
│   └── CMeEE-V2_test.json  # 测试集
├── CHIP-CTC/             # 临床筛选标准分类数据集
│   ├── CHIP-CTC_train.json # 训练集
│   ├── CHIP-CTC_dev.json   # 验证集
│   └── CHIP-CTC_test.json  # 测试集
├── models/               # 训练好的模型保存目录
├── results/              # 预测结果保存目录
├── scripts/              # 工具脚本目录
├── web/                  # Web应用目录
│   ├── web_app.py        # Web后端应用
│   └── templates/        # 前端模板
├── requirements.txt      # 依赖包列表
├── README.md             # 项目说明文档
├── README_WEB.md         # Web应用说明文档
└── report_outline.md     # 项目报告提纲
```

## 安装依赖

使用pip安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 依赖包说明

- python==3.12.4
- torch==2.2.0
- transformers==4.45.1
- numpy==1.26.4
- pandas==2.0.0
- scikit-learn==1.2.0
- flask==3.1.2
- pytorch-crf
- tqdm==4.67.1

## 使用方法

### 1. 医学实体抽取

#### 训练模型

```bash
# 首次训练（使用roberta模型，建议使用）
python src/main.py ee_train --data_dir CMeEE-V2 --model_name hfl/chinese-roberta-wwm-ext --batch_size 8 --epochs 10 --lr 1e-5 --output_dir ./models/entity_extraction

# 增量训练（继续训练已有模型）
python src/main.py ee_train --data_dir CMeEE-V2 --model_name hfl/chinese-roberta-wwm-ext --batch_size 8 --epochs 5 --lr 5e-6 --output_dir ./models/entity_extraction
```

#### 预测结果

```bash
python src/main.py ee_predict --data_dir CMeEE-V2 --model_dir ./models/entity_extraction --output_file ./results/CMeEE-V2_test.json
```

### 2. 临床筛选标准分类

#### 训练模型

```bash
# 首次训练
python src/main.py tc_train --data_dir CHIP-CTC --model_name bert-base-chinese --batch_size 32 --epochs 5 --lr 2e-5 --output_dir ./models/text_classification

# 增量训练
python src/main.py tc_train --data_dir CHIP-CTC --model_name bert-base-chinese --batch_size 32 --epochs 3 --lr 1e-5 --output_dir ./models/text_classification
```

#### 预测结果

```bash
python src/main.py tc_predict --data_dir CHIP-CTC --model_dir ./models/text_classification --output_file ./results/CHIP-CTC_test.json
```

### 3. 启动Web应用

```bash
python web/web_app.py
```

启动后，在浏览器中访问：
```
http://127.0.0.1:5000
```

## 模型说明

### 医学实体抽取

采用**BERT + BiLSTM + CRF**的强大模型架构：
- **BERT层**：使用hfl/chinese-roberta-wwm-ext预训练模型，捕获上下文语义信息
- **BiLSTM层**：进一步提取序列特征，增强模型对长距离依赖的建模能力
- **CRF层**：确保实体标签的一致性，提高序列标注的准确性
- **对抗训练（FGM）**：增强模型的泛化能力和鲁棒性

### 临床筛选标准分类

使用**BertForSequenceClassification**进行文本分类：
- 基于BERT预训练模型，直接输出44种语义类别的概率分布
- 采用**Focal Loss**替代交叉熵损失函数，解决类别不平衡问题
- 使用**余弦退火学习率调度器**优化训练过程
- 添加**对抗训练（FGM）**，增强模型的泛化能力和鲁棒性
- 基于**Macro-F1**的模型保存策略，保留最佳模型
- 支持增量训练和模型保存

### 增量训练支持

系统支持增量训练，每次训练前会检查指定目录是否存在已有模型：
- 如果存在模型，自动加载并继续训练
- 如果不存在模型，初始化新模型开始训练
- 支持调整学习率、批次大小等参数进行微调

## 评估指标

1. **医学实体抽取**：采用Micro-F1作为主评测指标
2. **临床筛选标准分类**：采用Macro-F1作为主评测指标

## 结果提交

### 医学实体抽取

预测结果保存为JSON文件，格式参考`CMeEE-V2/example_pred.json`，每个样本格式为：
```json
{
    "text": "文本内容",
    "entities": [
        {
            "start_idx": 起始位置,
            "end_idx": 结束位置,
            "entity": "实体文本",
            "type": "实体类型"
        }
    ]
}
```

### 临床筛选标准分类

预测结果保存为JSON文件，格式参考`CHIP-CTC/example_pred.json`，每个样本格式为：
```json
{
    "id": "样本ID",
    "text": "文本内容",
    "label": "预测类别"
}
```

## 注意事项

1. **预训练模型下载**：首次运行时会自动下载预训练模型，需要联网。建议使用hfl/chinese-roberta-wwm-ext模型，在中文任务上表现更好。
2. **GPU内存要求**：
   - 使用RoBERTa模型时，建议GPU内存不小于8GB
   - batch_size=8时，GPU内存占用约4-6GB
   - 如内存不足，可进一步减小batch_size
3. **训练时长**：
   - 完整训练10轮约需要2-3小时（取决于GPU性能）
   - 对抗训练会增加约20%的训练时间，但能显著提高模型性能
4. **增量训练注意事项**：
   - 确保输出目录存在且包含有效的模型文件
   - 增量训练时建议降低学习率（如从1e-5降至5e-6）
   - 可调整批次大小和训练轮数进行微调
5. **模型加载问题**：
   - 如遇到模型加载错误，可尝试使用新的输出目录重新训练
   - 确保transformers库版本与训练时一致
6. **参数调整建议**：
   - 学习率：建议范围1e-5 ~ 5e-6
   - 批次大小：建议范围8 ~ 32
   - 训练轮数：建议范围5 ~ 15
7. **Web应用支持响应式设计**：适配多种设备，可在浏览器中实时使用
8. **结果提交格式**：确保预测结果格式符合平台要求，系统会自动生成正确格式

## 技术亮点

1. **强大的模型架构**：医学实体抽取采用BERT + BiLSTM + CRF架构，结合对抗训练，提高模型性能
2. **增量训练支持**：每次训练自动检查已有模型，支持模型的持续优化
3. **优化的训练策略**：
   - 使用余弦退火学习率调度器
   - 支持早停机制，防止过拟合
   - 基于损失和F1-score的双指标模型保存
4. **高效的推理速度**：预测速度可达65+样本/秒
5. **完善的结果生成机制**：自动生成符合平台要求的提交格式
6. **支持多种预训练模型**：可灵活切换不同的BERT或RoBERTa模型
7. **完整的前后端架构**：RESTful API设计，易于扩展
8. **优秀的用户界面设计**：基于格式塔心理学和菲茨定律的设计原则

## 联系方式

如有问题，请联系项目负责人。
