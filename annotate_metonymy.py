import json
import os
import sys
import time
import concurrent.futures
from tqdm import tqdm

# 添加父目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metonymy.metonymy_annotator import MetonymyAnnotator
from src.data.api_utils import get_qwen_api_key

# 定义路径
base_dir = r"d:\YZQ\PAPER\paper\多模态隐喻\多模态推荐中的情感转喻建模\MetonymyRec_master_yzq\data\Yelp"
input_file = os.path.join(base_dir, "structured_yelp_data.json")
output_file = os.path.join(base_dir, "structured_yelp_data_with_metonymy.json")

# 生成图片完整路径
def get_image_path(image):
    """获取图片的完整路径"""
    if 'image_path' in image:
        # 如果已经有image_path字段，使用它
        img_path = image['image_path']
        # 构建完整路径
        base_image_dir = r"d:\YZQ\PAPER\paper\多模态隐喻\多模态推荐中的情感转喻建模\MetonymyRec_master_yzq\data\Yelp\yelp_photos~"
        img_path = os.path.join(base_image_dir, img_path.replace('/', '\\'))
        return img_path
    return None

# 处理单个图片的转喻标注
def process_image(image, business_item, annotator):
    """处理单个图片的转喻标注"""
    # 获取图片路径
    image_path = get_image_path(image)
    if not image_path:
        return False, "找不到图片路径"
    
    if not os.path.exists(image_path):
        return False, f"图片文件不存在: {image_path}"
    
    # 获取评论文本（使用第一张评论作为代表）
    review_texts = business_item.get('review_text', [])
    tip_texts = business_item.get('tip_text', [])
    
    # 优先使用评论，然后使用二次评论
    if review_texts and review_texts[0] != "No reviews available":
        review_text = review_texts[0]
    elif tip_texts and tip_texts[0] != "No tips available":
        review_text = tip_texts[0]
    else:
        review_text = image.get('caption', '')
    
    # 限制评论文本长度
    if len(review_text) > 1000:
        review_text = review_text[:1000] + "..."
    
    try:
        # 标注转喻
        annotation = annotator.annotate(
            review_text=review_text,
            image_path=image_path
        )
        # 将标注结果添加到图片信息中
        image['metonymy_annotation'] = annotation
        return True, "标注成功"
    except Exception as e:
        return False, f"标注失败: {e}"

# 主函数
def main():
    print("读取JSON文件...")
    
    # 分批读取数据，减少内存使用
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"读取到 {len(data)} 条商家数据")
    
    # 只处理前5条数据（用于测试）
    data = data[:5]
    total_data = len(data)
    print(f"处理前 {total_data} 条数据")
    
    # 初始化标注器
    # 加载API密钥
    api_key = get_qwen_api_key()
    
    if not api_key:
        raise ValueError("错误：无法加载API密钥，请检查API密钥配置")
    
    print(f"API密钥: {api_key[:10]}...")
    
    annotator = MetonymyAnnotator(api_key=api_key)
    
    # 处理每条数据
    processed_count = 0
    failed_count = 0
    
    start_time = time.time()
    
    for i, business_item in enumerate(data):
        print(f"\n处理第 {i+1}/{total_data} 条商家数据: {business_item.get('business_name', 'Unknown')}")
        
        # 获取图片列表
        images = business_item.get('images', [])
        if not images:
            print("  警告：该商家没有图片")
            continue
        
        # 对每张图片进行标注
        for j, image in enumerate(images):
            print(f"  处理第 {j+1}/{len(images)} 张图片...")
            
            # 处理单个图片
            success, message = process_image(image, business_item, annotator)
            print(f"  {message}")
            
            if success:
                processed_count += 1
            else:
                failed_count += 1
            
            # 添加短暂延迟，避免API调用过于频繁
            time.sleep(1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 保存更新后的数据
    print(f"\n保存更新后的数据到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("完成！")
    print(f"总处理时间: {elapsed_time:.2f} 秒")
    print(f"成功处理: {processed_count} 张图片")
    print(f"失败处理: {failed_count} 张图片")

if __name__ == "__main__":
    main()