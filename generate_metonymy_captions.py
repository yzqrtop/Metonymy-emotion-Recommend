import json
import os
import argparse
import torch
import time
from PIL import Image
from tqdm import tqdm

# 添加父目录到Python路径
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metonymy.metonymy_captioner import MetonymyAwareCaptioner

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

# 处理单个图片的字幕生成
def process_image(image, business_item, captioner):
    """处理单个图片的字幕生成"""
    # 获取图片路径
    img_path = get_image_path(image)
    if not img_path:
        return False, "图片没有路径信息"
    
    if not os.path.exists(img_path):
        return False, f"图片文件不存在: {img_path}"
    
    # 提取评论文本
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
    
    # 提取转喻标注信息（如果存在）
    metonymy_annotation = image.get('metonymy_annotation', {})
    metonymy_type = metonymy_annotation.get('metonymy_type', "unknown")
    inferred_sentiment = metonymy_annotation.get('inferred_sentiment', 'neutral')
    
    try:
        # 打开图片
        image_obj = Image.open(img_path)
        
        # 生成字幕（传入转喻类型和情感信息）
        caption = captioner.generate_caption(
            image_obj, 
            review_text=review_text,
            metonymy_type=metonymy_type,
            sentiment=inferred_sentiment
        )
        # 将字幕更新到图片信息中
        image['metonymy_caption'] = caption
        return True, "字幕生成成功"
    except Exception as e:
        return False, f"字幕生成失败: {e}"

def generate_metonymy_captions(input_file, output_file):
    """
    生成转喻感知字幕并更新到数据中
    
    Args:
        input_file: 输入数据文件路径
        output_file: 输出数据文件路径
    """
    print(f"正在读取输入文件: {input_file}")
    
    # 读取输入数据（JSON格式）
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"读取到 {len(data)} 条商家数据")
    
    # 只处理前5条数据（用于测试）
    data = data[:5]
    print(f"只处理前 {len(data)} 条数据")
    
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 初始化字幕生成器
    # 使用Qwen3-VL模型
    captioner = MetonymyAwareCaptioner(
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        device=device,
        local_cache_dir="./models/qwen3",
        timeout=5,
        model_provider="local"
    )
    
    # 处理每条数据
    processed_count = 0
    failed_count = 0
    
    start_time = time.time()
    
    for i, business_item in enumerate(data):
        print(f"\n处理第 {i+1}/{len(data)} 条商家数据: {business_item.get('business_name', 'Unknown')}")
        
        # 获取图片列表
        images = business_item.get('images', [])
        if not images:
            print("  警告：该商家没有图片")
            continue
        
        # 对每张图片生成字幕
        for j, image in enumerate(images):
            print(f"  处理第 {j+1}/{len(images)} 张图片...")
            
            # 处理单个图片
            success, message = process_image(image, business_item, captioner)
            print(f"  {message}")
            
            if success:
                processed_count += 1
            else:
                failed_count += 1
            
            # 添加短暂延迟，避免处理过于频繁
            time.sleep(0.5)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 保存更新后的数据
    print(f"\n正在保存输出文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成！")
    print(f"总处理时间: {elapsed_time:.2f} 秒")
    print(f"成功处理: {processed_count} 张图片")
    print(f"失败处理: {failed_count} 张图片")
    print(f"已为 {len(data)} 条数据生成转喻感知字幕")
    print(f"更新后的数据已保存到: {output_file}")


if __name__ == "__main__":
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成转喻感知字幕")
    parser.add_argument("--input-file", default=r"d:\YZQ\PAPER\paper\多模态隐喻\多模态推荐中的情感转喻建模\MetonymyRec_master_yzq\data\Yelp\structured_yelp_data_with_metonymy.json", help="输入数据文件路径")
    parser.add_argument("--output-file", default=r"d:\YZQ\PAPER\paper\多模态隐喻\多模态推荐中的情感转喻建模\MetonymyRec_master_yzq\data\Yelp\structured_yelp_data_with_captions.json", help="输出数据文件路径")
    
    args = parser.parse_args()
    
    # 执行字幕生成
    generate_metonymy_captions(args.input_file, args.output_file)