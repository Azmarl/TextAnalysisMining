import json
import os
from typing import List, Dict, Tuple

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def load_cmeeev2_data(self, split: str) -> List[Dict]:
        file_map = {
            'train': 'CMeEE-V2_train.json',
            'dev': 'CMeEE-V2_dev.json',
            'test': 'CMeEE-V2_test.json'
        }
        file_path = os.path.join(self.data_dir, file_map[split])
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def load_chip_ctc_data(self, split: str) -> List[Dict]:
        file_map = {
            'train': 'CHIP-CTC_train.json',
            'dev': 'CHIP-CTC_dev.json',
            'test': 'CHIP-CTC_test.json'
        }
        file_path = os.path.join(self.data_dir, file_map[split])
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def load_category_mapping(self) -> Dict[str, int]:
        # 44类定义
        categories = [
            'Disease', 'Symptom', 'Sign', 'Pregnancy-related Activity', 'Neoplasm Status', 'Non-Neoplasm Disease Stage', 'Allergy Intolerance',
            'Organ or Tissue Status', 'Life Expectancy', 'Oral related', 'Pharmaceutical Substance or Drug', 'Therapy or Surgery',
            'Device', 'Nursing', 'Diagnostic', 'Laboratory Examinations',
            'Risk Assessment', 'Receptor Status', 'Age', 'Special Patient Characteristic',
            'Literacy', 'Gender', 'Education', 'Address',
            'Ethnicity', 'Consent', 'Enrollment in other studies',
            'Researcher Decision', 'Capacity', 'Ethical Audit', 'Compliance with Protocol', 'Addictive Behavior',
            'Bedtime', 'Exercise', 'Diet', 'Alcohol Consumer',
            'Sexual related', 'Smoking Status', 'Blood Donation', 'Encounter',
            'Disabilities', 'Healthy', 'Data Accessible', 'Multiple'
        ]
        return {cat: idx for idx, cat in enumerate(categories)}

    def convert_cmeeev2_to_bio(self, data: List[Dict]) -> List[Tuple[str, List[str]]]:
        """
        [关键优化] 将CMeEE-V2数据转换为BIO标注格式
        采用“最长实体优先”策略来缓解嵌套实体问题
        """
        bio_data = []
        
        for item in data:
            text = item['text']
            entities = item.get('entities', [])
            
            # 初始化全是 'O'
            labels = ['O'] * len(text)
            
            # [关键步骤] 按照实体长度降序排序
            # 这样长的实体会先占据位置，避免被短实体截断或覆盖
            if entities:
                sorted_entities = sorted(entities, key=lambda x: (x['end_idx'] - x['start_idx']), reverse=True)
                
                for entity in sorted_entities:
                    start = entity['start_idx']
                    end = entity['end_idx']
                    entity_type = entity['type']
                    
                    # 检查该区间是否已经被占用（只要有一个字不为O，就算占用）
                    # 注意：BIO策略下，无法完美处理嵌套，这里选择保留最长实体
                    is_occupied = False
                    for i in range(start, end): # end_idx 在 CMeEE 是开区间还是闭区间需注意，通常是左闭右开 [start, end)
                         # CMeEE 数据样例中: "start_idx": 3, "end_idx": 8, "entity": "房室结消融" (5个字)
                         # 说明是左闭右开区间 range(3, 8) -> 3,4,5,6,7
                        if i < len(labels) and labels[i] != 'O':
                            is_occupied = True
                            break
                    
                    if not is_occupied:
                        if start < len(labels):
                            labels[start] = f'B-{entity_type}'
                        for i in range(start + 1, end):
                            if i < len(labels):
                                labels[i] = f'I-{entity_type}'
            
            bio_data.append((text, labels))
        
        return bio_data