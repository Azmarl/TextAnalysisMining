import os
from typing import List, Dict, Set
from collections import defaultdict

class EntityExtractionEvaluator:
    """
    医学实体抽取评估器
    """
    
    def __init__(self):
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
    
    def compute_metrics(self, true_entities: List[List[Dict]], pred_entities: List[List[Dict]]) -> Dict[str, float]:
        """
        计算实体抽取的Micro-F1指标
        Args:
            true_entities: 真实实体列表，每个元素是一个文本的实体列表
            pred_entities: 预测实体列表，每个元素是一个文本的实体列表
        Returns:
            包含precision, recall, f1的字典
        """
        true_positive = 0
        false_positive = 0
        false_negative = 0
        
        for true_list, pred_list in zip(true_entities, pred_entities):
            # 将实体转换为集合，便于比较
            true_set = set()
            pred_set = set()
            
            for entity in true_list:
                true_set.add((entity['start_idx'], entity['end_idx'], entity['type']))
            
            for entity in pred_list:
                pred_set.add((entity['start_idx'], entity['end_idx'], entity['type']))
            
            # 计算TP, FP, FN
            true_positive += len(true_set & pred_set)
            false_positive += len(pred_set - true_set)
            false_negative += len(true_set - pred_set)
        
        # 计算precision, recall, f1
        if true_positive + false_positive == 0:
            precision = 0.0
        else:
            precision = true_positive / (true_positive + false_positive)
        
        if true_positive + false_negative == 0:
            recall = 0.0
        else:
            recall = true_positive / (true_positive + false_negative)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def print_results(self):
        """
        打印评估结果
        """
        print(f'Precision: {self.precision:.4f}')
        print(f'Recall: {self.recall:.4f}')
        print(f'F1: {self.f1:.4f}')

class TextClassificationEvaluator:
    """
    文本分类评估器
    """
    
    def __init__(self):
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.macro_precision = 0.0
        self.macro_recall = 0.0
        self.macro_f1 = 0.0
    
    def compute_metrics(self, true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
        """
        计算文本分类的Macro-F1指标
        Args:
            true_labels: 真实标签列表
            pred_labels: 预测标签列表
        Returns:
            包含precision, recall, f1, macro_precision, macro_recall, macro_f1的字典
        """
        # 计算Micro-F1
        true_positive = 0
        false_positive = 0
        false_negative = 0
        
        # 计算每个类别的TP, FP, FN
        class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for true, pred in zip(true_labels, pred_labels):
            if true == pred:
                true_positive += 1
                class_metrics[true]['tp'] += 1
            else:
                false_positive += 1
                false_negative += 1
                class_metrics[pred]['fp'] += 1
                class_metrics[true]['fn'] += 1
        
        # 计算Micro指标
        total = true_positive + false_positive
        if total == 0:
            micro_precision = 0.0
        else:
            micro_precision = true_positive / total
        
        total = true_positive + false_negative
        if total == 0:
            micro_recall = 0.0
        else:
            micro_recall = true_positive / total
        
        if micro_precision + micro_recall == 0:
            micro_f1 = 0.0
        else:
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        
        # 计算Macro指标
        class_precisions = []
        class_recalls = []
        class_f1s = []
        
        for cls, metrics in class_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            if tp + fp == 0:
                cls_precision = 0.0
            else:
                cls_precision = tp / (tp + fp)
            
            if tp + fn == 0:
                cls_recall = 0.0
            else:
                cls_recall = tp / (tp + fn)
            
            if cls_precision + cls_recall == 0:
                cls_f1 = 0.0
            else:
                cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall)
            
            class_precisions.append(cls_precision)
            class_recalls.append(cls_recall)
            class_f1s.append(cls_f1)
        
        if not class_precisions:
            macro_precision = 0.0
            macro_recall = 0.0
            macro_f1 = 0.0
        else:
            macro_precision = sum(class_precisions) / len(class_precisions)
            macro_recall = sum(class_recalls) / len(class_recalls)
            macro_f1 = sum(class_f1s) / len(class_f1s)
        
        self.precision = micro_precision
        self.recall = micro_recall
        self.f1 = micro_f1
        self.macro_precision = macro_precision
        self.macro_recall = macro_recall
        self.macro_f1 = macro_f1
        
        return {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }
    
    def print_results(self):
        """
        打印评估结果
        """
        print(f'Micro-Precision: {self.precision:.4f}')
        print(f'Micro-Recall: {self.recall:.4f}')
        print(f'Micro-F1: {self.f1:.4f}')
        print(f'Macro-Precision: {self.macro_precision:.4f}')
        print(f'Macro-Recall: {self.macro_recall:.4f}')
        print(f'Macro-F1: {self.macro_f1:.4f}')

class ResultSaver:
    """
    结果保存器
    """
    
    @staticmethod
    def save_entity_extraction_results(results: List[List[Dict]], output_path: str, original_data: List[Dict] = None):
        """
        保存实体抽取结果到文件（支持两种格式：JSON格式用于提交，TXT格式用于本地查看）
        Args:
            results: 预测实体列表，每个元素是一个文本的实体列表
            output_path: 输出文件路径
            original_data: 原始测试数据，包含id和text字段（用于生成JSON格式）
        """
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 检查输出文件格式
        if output_path.endswith('.json'):
            # JSON格式（用于提交）
            if original_data is None:
                raise ValueError("保存JSON格式需要提供original_data参数")
            
            # 构建提交格式
            submission_data = []
            for item, entities in zip(original_data, results):
                # 确保实体字段存在
                if 'entities' not in item:
                    item['entities'] = []
                
                # 按照start_idx排序
                sorted_entities = sorted(entities, key=lambda x: x['start_idx'])
                
                submission_item = {
                    'text': item['text'],
                    'entities': sorted_entities
                }
                # 如果有id字段，也添加进去
                if 'id' in item:
                    submission_item['id'] = item['id']
                
                submission_data.append(submission_item)
            
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(submission_data, f, ensure_ascii=False, indent=2)
        else:
            # TXT格式（用于本地查看）
            with open(output_path, 'w', encoding='utf-8') as f:
                for entities in results:
                    if not entities:
                        f.write('\n')
                        continue
                    
                    # 按照start_idx排序
                    entities = sorted(entities, key=lambda x: x['start_idx'])
                    
                    # 格式化输出
                    entity_strs = []
                    for entity in entities:
                        entity_str = f"{entity['start_idx']} {entity['end_idx']} {entity['entity']} {entity['type']}"
                        entity_strs.append(entity_str)
                    
                    f.write('\t'.join(entity_strs) + '\n')
    
    @staticmethod
    def save_text_classification_results(data: List[Dict], pred_labels: List[str], output_path: str):
        """
        保存文本分类结果到文件
        Args:
            data: 原始数据列表
            pred_labels: 预测标签列表
            output_path: 输出文件路径
        """
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results = []
        for item, pred_label in zip(data, pred_labels):
            result = {
                'id': item['id'],
                'text': item['text'],
                'label': pred_label
            }
            results.append(result)
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
