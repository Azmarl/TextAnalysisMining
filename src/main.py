import os
import sys
import argparse
from typing import List, Dict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader
from src.models.entity_extraction import EntityExtractionModel
from src.models.text_classification import TextClassificationModel
from src.evaluation.evaluator import (
    EntityExtractionEvaluator,
    TextClassificationEvaluator,
    ResultSaver
)

def train_entity_extraction(args):
    """
    训练医学实体抽取模型
    """
    print(f"开始训练医学实体抽取模型... 使用模型: {args.model_name}")
    
    # 加载数据
    data_loader = DataLoader(args.data_dir)
    train_data = data_loader.load_cmeeev2_data('train')
    dev_data = data_loader.load_cmeeev2_data('dev')
    
    # 转换为BIO格式 (现在使用了长实体优先策略)
    train_bio = data_loader.convert_cmeeev2_to_bio(train_data)
    dev_bio = data_loader.convert_cmeeev2_to_bio(dev_data)
    
    # 检查是否存在已有模型，支持增量训练
    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        print(f"加载已有模型: {args.output_dir}")
        model = EntityExtractionModel()
        model.load_model(args.output_dir)
    else:
        print(f"初始化新模型: {args.model_name}")
        model = EntityExtractionModel(model_name=args.model_name)
    
    # 训练模型 (所有的 Loop 和 FGM 都在这里面)
    # dev_data_raw 用于可能的评估，这里主要依靠 BIO 数据训练
    model.train_model(
        train_bio, 
        dev_bio, 
        dev_data_raw=dev_data, 
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        lr=args.lr, 
        output_dir=args.output_dir
    )
    
    # 训练结束后进行一次评估
    print("训练结束，进行最终评估...")
    evaluator = EntityExtractionEvaluator()
    true_entities = [item.get('entities', []) for item in dev_data]
    pred_entities = []
    
    # 预测过程
    for item in dev_data: 
        text = item['text']
        pred = model.predict(text)
        pred_entities.append(pred)
    
    metrics = evaluator.compute_metrics(true_entities, pred_entities)
    evaluator.print_results()

def predict_entity_extraction(args):
    print("开始预测医学实体抽取...")
    data_loader = DataLoader(args.data_dir)
    test_data = data_loader.load_cmeeev2_data('test')
    
    model = EntityExtractionModel()
    model.load_model(args.model_dir)
    
    pred_entities = []
    # 添加进度条
    from tqdm import tqdm
    for item in tqdm(test_data):
        text = item['text']
        pred = model.predict(text)
        pred_entities.append(pred)
    
    ResultSaver.save_entity_extraction_results(pred_entities, args.output_file, test_data)
    print(f"结果保存到: {args.output_file}")

def train_text_classification(args):
    """
    训练临床筛选标准分类模型
    """
    print("开始训练临床筛选标准分类模型...")
    
    # 加载数据
    data_loader = DataLoader(args.data_dir)
    train_data = data_loader.load_chip_ctc_data('train')
    dev_data = data_loader.load_chip_ctc_data('dev')
    
    # 加载类别映射
    category_map = data_loader.load_category_mapping()
    
    # 初始化模型
    model = TextClassificationModel(model_name=args.model_name, num_labels=len(category_map))
    model.set_category_mapping(category_map)
    
    # 训练模型
    model.train(train_data, dev_data, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)
    
    # 保存模型
    model.save_model(args.output_dir)
    print(f"模型保存到: {args.output_dir}")
    
    # 评估模型
    print("评估模型...")
    evaluator = TextClassificationEvaluator()
    
    # 获取验证集的真实标签和预测标签
    true_labels = [item['label'] for item in dev_data]
    texts = [item['text'] for item in dev_data]
    pred_labels = model.predict_batch(texts)
    
    # 计算指标
    metrics = evaluator.compute_metrics(true_labels, pred_labels)
    evaluator.print_results()

def predict_text_classification(args):
    """
    预测临床筛选标准分类结果
    """
    print("开始预测临床筛选标准分类...")
    
    # 加载数据
    data_loader = DataLoader(args.data_dir)
    test_data = data_loader.load_chip_ctc_data('test')
    
    # 加载类别映射
    category_map = data_loader.load_category_mapping()
    
    # 加载模型
    model = TextClassificationModel(num_labels=len(category_map))
    model.load_model(args.model_dir)
    model.set_category_mapping(category_map)
    
    # 预测
    texts = [item['text'] for item in test_data]
    pred_labels = model.predict_batch(texts)
    
    # 保存结果
    ResultSaver.save_text_classification_results(test_data, pred_labels, args.output_file)
    print(f"结果保存到: {args.output_file}")

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="医学文本挖掘系统")
    subparsers = parser.add_subparsers(dest="task", help="任务类型")
    
    # 医学实体抽取训练
    ee_train_parser = subparsers.add_parser("ee_train", help="训练医学实体抽取模型")
    ee_train_parser.add_argument("--data_dir", type=str, default=".", help="数据目录")
    ee_train_parser.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext", help="预训练模型")
    ee_train_parser.add_argument("--batch_size", type=int, default=16, help="批次大小 (RoBERTa 较大，建议减小)")
    ee_train_parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    ee_train_parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    ee_train_parser.add_argument("--output_dir", type=str, default="./models/entity_extraction", help="模型输出目录")
    
    # EE Predict
    ee_pred_parser = subparsers.add_parser("ee_predict", help="预测")
    ee_pred_parser.add_argument("--data_dir", type=str, default=".", help="数据目录")
    ee_pred_parser.add_argument("--model_dir", type=str, required=True, help="模型目录")
    ee_pred_parser.add_argument("--output_file", type=str, default="./results/entity_extraction_pred.json", help="输出文件") 
    
    # 临床筛选标准分类训练
    tc_train_parser = subparsers.add_parser("tc_train", help="训练临床筛选标准分类模型")
    tc_train_parser.add_argument("--data_dir", type=str, default=".", help="数据目录")
    tc_train_parser.add_argument("--model_name", type=str, default="bert-base-chinese", help="预训练模型名称")
    tc_train_parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    tc_train_parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    tc_train_parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    tc_train_parser.add_argument("--output_dir", type=str, default="./models/text_classification", help="模型输出目录")
    
    # 临床筛选标准分类预测
    tc_pred_parser = subparsers.add_parser("tc_predict", help="预测临床筛选标准分类结果")
    tc_pred_parser.add_argument("--data_dir", type=str, default=".", help="数据目录")
    tc_pred_parser.add_argument("--model_dir", type=str, required=True, help="模型目录")
    tc_pred_parser.add_argument("--output_file", type=str, default="./results/text_classification_pred.json", help="预测结果输出文件")
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    args = parse_args()
    
    if args.task == "ee_train":
        train_entity_extraction(args)
    elif args.task == "ee_predict":
        predict_entity_extraction(args)
    elif args.task == "tc_train":
        train_text_classification(args)
    elif args.task == "tc_predict":
        predict_text_classification(args)
    else:
        print("请指定任务类型，可选：ee_train, ee_predict, tc_train, tc_predict")

if __name__ == "__main__":
    main()
