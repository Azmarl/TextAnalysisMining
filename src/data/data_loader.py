import json
import os
from typing import List, Dict, Tuple

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def load_cmeeev2_data(self, split: str) -> List[Dict]:
        """
        加载CMeEE-V2数据集
        Args:
            split: 数据集分割，可选值：train, dev, test
        Returns:
            数据列表，每个元素包含text和entities字段
        """
        file_map = {
            'train': 'CMeEE-V2_train.json',
            'dev': 'CMeEE-V2_dev.json',
            'test': 'CMeEE-V2_test.json'
        }
        
        file_path = os.path.join(self.data_dir, 'CMeEE-V2', file_map[split])
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def load_chip_ctc_data(self, split: str) -> List[Dict]:
        """
        加载CHIP-CTC数据集
        Args:
            split: 数据集分割，可选值：train, dev, test
        Returns:
            数据列表，每个元素包含id, text和label字段
        """
        file_map = {
            'train': 'CHIP-CTC_train.json',
            'dev': 'CHIP-CTC_dev.json',
            'test': 'CHIP-CTC_test.json'
        }
        
        file_path = os.path.join(self.data_dir, 'CHIP-CTC', file_map[split])
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def preprocess_text(self, text: str) -> str:
        """
        文本预处理
        Args:
            text: 原始文本
        Returns:
            预处理后的文本
        """
        # 去除首尾空格
        text = text.strip()
        # 替换多余的空格为单个空格
        text = ' '.join(text.split())
        return text
    
    def convert_cmeeev2_to_bio(self, data: List[Dict]) -> List[Tuple[str, List[str]]]:
        """
        将CMeEE-V2数据转换为BIO标注格式
        Args:
            data: CMeEE-V2原始数据
        Returns:
            转换后的数据，每个元素为(text, labels)元组
        """
        bio_data = []
        
        for item in data:
            text = item['text']
            entities = item.get('entities', [])
            
            # 初始化标签列表
            labels = ['O'] * len(text)
            
            # 处理每个实体
            for entity in entities:
                start = entity['start_idx']
                end = entity['end_idx']
                entity_type = entity['type']
                
                # 设置B标签
                labels[start] = f'B-{entity_type}'
                # 设置I标签
                for i in range(start + 1, end + 1):
                    if i < len(labels):
                        labels[i] = f'I-{entity_type}'
            
            bio_data.append((text, labels))
        
        return bio_data
    
    def load_category_mapping(self) -> Dict[str, int]:
        """
        加载分类映射
        Returns:
            类别到索引的映射
        """
        # 这里直接定义44个类别，实际使用时可以从category.xlsx加载
        categories = [
            'Age', 'Gender', 'Race', 'Ethnicity', 'Smoking', 'Alcohol', 'Pregnancy',
            'Breast Feeding', 'Contraception', 'Menopause', 'Weight', 'Height',
            'BMI', 'Vital Signs', 'Physical Examination', 'Laboratory Tests',
            'Imaging Studies', 'Electrocardiogram', 'Endoscopy', 'Biopsy',
            'Pathology', 'Genetics', 'Family History', 'Personal History',
            'Medical History', 'Surgical History', 'Medication History',
            'Allergy History', 'Sign', 'Symptom', 'Disease', 'Therapy or Surgery',
            'Radiation Therapy', 'Chemotherapy', 'Immunotherapy', 'Targeted Therapy',
            'Stem Cell Transplant', 'Blood Transfusion', 'Dialysis', 'Organ Transplant',
            'Palliative Care', 'Hospice Care', 'Other Therapy', 'Other'
        ]
        
        return {cat: idx for idx, cat in enumerate(categories)}
