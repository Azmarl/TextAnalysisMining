# 医学文本挖掘系统 - Web应用

## 1. 功能介绍

本Web应用提供了一个交互式界面，用于展示和使用医学文本挖掘系统的两个核心功能：

### 1.1 医学实体抽取
- 从医学文本中识别9种医学实体
- 高亮显示实体位置
- 展示实体类型和详细信息
- 支持实时预测

### 1.2 临床文本分类
- 对临床筛选标准进行44分类
- 显示预测类别
- 支持实时预测

## 2. 技术架构

- **后端**：Flask 3.1.2
- **前端**：HTML + CSS + JavaScript + Bootstrap 5
- **模型**：基于BERT的深度学习模型

## 3. 安装和运行

### 3.1 安装依赖

确保已经安装了所有必要的依赖：

```bash
D:\VsCode\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3.2 启动Web服务

```bash
D:\VsCode\.venv\Scripts\python.exe web_app.py
```

### 3.3 访问应用

启动成功后，在浏览器中访问：
```
http://127.0.0.1:5000
```

## 4. 使用说明

### 4.1 医学实体抽取

1. 在"医学实体抽取"标签页中输入医学文本
2. 点击"实体抽取"按钮
3. 查看结果：
   - 高亮显示的文本
   - 实体标签
   - 详细的实体信息表格

### 4.2 临床文本分类

1. 在"临床文本分类"标签页中输入临床筛选标准
2. 点击"文本分类"按钮
3. 查看预测的类别

## 5. 模型说明

Web应用使用的模型与命令行版本相同：

- **实体抽取模型**：BERT-base-chinese + TokenClassification
- **文本分类模型**：BERT-base-chinese + SequenceClassification

模型文件需要预先训练并保存到以下目录：
- 实体抽取模型：`./models/entity_extraction/`
- 文本分类模型：`./models/text_classification/`

## 6. 训练模型

如果尚未训练模型，可以使用以下命令进行训练：

### 6.1 训练实体抽取模型

```bash
D:\VsCode\.venv\Scripts\python.exe src/main.py ee_train --data_dir . --model_name bert-base-chinese --batch_size 32 --epochs 5 --lr 2e-5 --output_dir ./models/entity_extraction
```

### 6.2 训练文本分类模型

```bash
D:\VsCode\.venv\Scripts\python.exe src/main.py tc_train --data_dir . --model_name bert-base-chinese --batch_size 32 --epochs 5 --lr 2e-5 --output_dir ./models/text_classification
```

## 7. 接口说明

### 7.1 实体抽取接口

- **URL**：`/predict_entity`
- **方法**：`POST`
- **请求格式**：
  ```json
  {
    "text": "患者有糖尿病，出现了尿频、尿急的症状。"
  }
  ```
- **响应格式**：
  ```json
  {
    "success": true,
    "text": "患者有糖尿病，出现了尿频、尿急的症状。",
    "entities": [
      {
        "start_idx": 4,
        "end_idx": 6,
        "entity": "糖尿病",
        "type": "dis"
      },
      {
        "start_idx": 11,
        "end_idx": 12,
        "entity": "尿频",
        "type": "sym"
      },
      {
        "start_idx": 14,
        "end_idx": 15,
        "entity": "尿急",
        "type": "sym"
      }
    ]
  }
  ```

### 7.2 文本分类接口

- **URL**：`/predict_text`
- **方法**：`POST`
- **请求格式**：
  ```json
  {
    "text": "18岁以下的患者"
  }
  ```
- **响应格式**：
  ```json
  {
    "success": true,
    "text": "18岁以下的患者",
    "label": "Age"
  }
  ```

## 8. 注意事项

1. 首次启动时，模型加载需要一定时间
2. 确保模型文件存在于指定目录
3. 建议使用Chrome或Firefox浏览器访问
4. 文本输入不要过长，建议不超过512字符

## 9. 扩展开发

### 9.1 添加新的实体类型

1. 修改`src/models/entity_extraction.py`中的标签映射
2. 重新训练模型
3. 更新前端的CSS样式（如果需要）

### 9.2 添加新的分类类别

1. 修改`category.xlsx`文件
2. 重新训练模型
3. 更新前端显示（如果需要）

## 10. 联系方式

如有问题或建议，请联系项目负责人。
