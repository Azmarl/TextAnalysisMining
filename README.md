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

- torch==2.2.0
- transformers==4.30.0
- numpy==1.24.0
- pandas==2.0.0
- scikit-learn==1.2.0
- flask==3.1.2

## 使用方法

### 1. 医学实体抽取

#### 训练模型

```bash
python src/main.py ee_train --data_dir . --model_name bert-base-chinese --batch_size 32 --epochs 5 --lr 2e-5 --output_dir ./models/entity_extraction
```

#### 预测结果

```bash
python src/main.py ee_predict --data_dir . --model_dir ./models/entity_extraction --output_file ./results/CMeEE-V2_test.json
```

### 2. 临床筛选标准分类

#### 训练模型

```bash
python src/main.py tc_train --data_dir . --model_name bert-base-chinese --batch_size 32 --epochs 5 --lr 2e-5 --output_dir ./models/text_classification
```

#### 预测结果

```bash
python src/main.py tc_predict --data_dir . --model_dir ./models/text_classification --output_file ./results/CHIP-CTC_test.json
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

本项目采用BERT预训练模型作为基础模型，分别用于：

1. **医学实体抽取**：使用BertForTokenClassification进行序列标注
2. **临床筛选标准分类**：使用BertForSequenceClassification进行文本分类

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

1. 首次运行时会自动下载预训练模型，需要联网
2. 训练模型建议使用GPU支持，可加速训练过程
3. 可以根据实际情况调整batch_size和epochs参数
4. 可以尝试使用其他预训练模型，如roberta-base-chinese
5. Web应用支持响应式设计，适配多种设备

## 技术亮点

1. **高效的模型训练策略**：使用学习率调度和早停机制提高训练效率
2. **完善的结果生成机制**：自动生成符合平台要求的提交格式
3. **优秀的用户界面设计**：基于格式塔心理学和菲茨定律的设计原则
4. **完整的前后端架构**：RESTful API设计，易于扩展

## 联系方式

如有问题，请联系项目负责人。
