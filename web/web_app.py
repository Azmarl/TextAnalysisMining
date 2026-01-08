from flask import Flask, request, jsonify, render_template
import os
import sys
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.entity_extraction import EntityExtractionModel
from src.models.text_classification import TextClassificationModel
from src.data.data_loader import DataLoader

app = Flask(__name__, template_folder='templates')

# 全局变量，用于存储模型实例
entity_model = None
text_model = None

# 初始化模型
def init_models():
    global entity_model, text_model
    
    print("正在加载实体抽取模型...")
    entity_model = EntityExtractionModel()
    entity_model.load_model('./models/entity_extraction_new')
    print("实体抽取模型加载完成")
    
    print("正在加载文本分类模型...")
    text_model = TextClassificationModel()
    text_model.load_model('./models/text_classification_new')
    
    # 加载类别映射
    data_loader = DataLoader('.')
    category_map = data_loader.load_category_mapping()
    text_model.set_category_mapping(category_map)
    print("文本分类模型加载完成")

# 首页路由
@app.route('/')
def home():
    return render_template('index.html')

# 实体抽取预测接口
@app.route('/predict_entity', methods=['POST'])
def predict_entity():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': '文本不能为空'}), 400
        
        # 使用模型进行预测
        entities = entity_model.predict(text)
        
        return jsonify({
            'success': True,
            'text': text,
            'entities': entities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 文本分类预测接口
@app.route('/predict_text', methods=['POST'])
def predict_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': '文本不能为空'}), 400
        
        # 使用模型进行预测 (现在返回的是一个字典)
        result_dict = text_model.predict(text)
        
        return jsonify({
            'success': True,
            'text': text,
            'result': result_dict # 将整个结果字典传给前端
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 初始化模型
    init_models()
    
    # 启动Flask应用
    print("医学文本挖掘系统启动成功！")
    print("访问地址: http://127.0.0.1:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
