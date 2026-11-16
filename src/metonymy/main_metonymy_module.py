import os
import sys
import json
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.metonymy.metonymy_annotator import MetonymyAnnotator
from src.data.api_utils import get_qwen_api_key

def process_data(input_file, output_dir, api_key):
    """
    处理数据并进行转喻标注
    
    Args:
        input_file: 输入数据文件路径
        output_dir: 输出目录
        api_key: Qwen-VL API密钥
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取处理后的数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化标注器 - 使用与 test_metonymy_annotation.py 相同的模型
    annotator = MetonymyAnnotator(api_key=api_key, model="Qwen/Qwen3-VL-8B-Instruct")
    
    # 处理数据
    annotated_data = []
    
    for idx, item in tqdm(enumerate(data), total=len(data), desc="处理数据"):
        review_text = item.get('review_text', '')
        rating = item.get('rating', None)
        images = item.get('images', {})
        
        # 为每个类别处理图片
        for category, image_paths in images.items():
            if category in ['hi_res', 'large', 'thumb'] and isinstance(image_paths, list):
                for img_idx, image_path in enumerate(image_paths):
                    if image_path and os.path.exists(image_path):
                        try:
                            # 加上前置路径
                            if not image_path.startswith('./data/Yelp/Yelp_photos~/'):
                                image_path = './data/Yelp/Yelp_photos~/' + image_path
                            
                            # 进行转喻标注
                            annotation = annotator.annotate(
                                review_text=review_text,
                                image_path=image_path,
                                rating=rating
                            )
                            
                            # 构建标注结果
                            annotated_item = {
                                'review_text': review_text,
                                'rating': rating,
                                'image_category': category,
                                'image_index': img_idx,
                                'image_path': image_path,
                                'annotation': annotation
                            }
                            
                            annotated_data.append(annotated_item)
                            print(f"标注成功: 索引{idx}, 类别{category}, 图片{img_idx}")
                            
                        except Exception as e:
                            print(f"标注失败: 索引{idx}, 类别{category}, 图片{img_idx}: {e}")
    
    # 保存标注结果
    output_file = os.path.join(output_dir, 'annotated_data.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotated_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成！共标注 {len(annotated_data)} 条数据")
    print(f"标注结果已保存到: {output_file}")
    
    return annotated_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="转喻标注处理脚本")
    parser.add_argument("--input-file", default=r"d:\YZQ\PAPER\paper\多模态隐喻\多模态推荐中的情感转喻建模\MetonymyRec_master_yzq\src\test\processed_data.json", help="输入数据文件路径")
    parser.add_argument("--output-dir", default=r"d:\YZQ\PAPER\paper\多模态隐喻\多模态推荐中的情感转喻建模\MetonymyRec_master_yzq\data\output", help="输出目录")
    parser.add_argument("--api-key", help="API密钥（可选，默认从配置文件读取）")
    
    args = parser.parse_args()
    
    # 设置输入文件路径
    input_file = args.input_file
    
    # 设置输出目录
    output_dir = args.output_dir
    
    # 加载API密钥
    if args.api_key:
        api_key = args.api_key
    else:
        api_key = get_qwen_api_key()
    
    if not api_key:
        print("错误：无法加载API密钥，请检查API密钥配置")
        sys.exit(1)
    
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"API密钥: {api_key[:10]}..." if api_key else "未设置")
    
    # 处理数据
    annotated_data = process_data(
        input_file=input_file,
        output_dir=output_dir,
        api_key=api_key
    )
    
    print("\n标注完成！")