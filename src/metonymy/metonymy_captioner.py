import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import Optional, List
from modelscope import snapshot_download

# 尝试导入 Qwen2-VL 和 Qwen3-VL 相关的模型和处理器
try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
except ImportError as e:
    print(f"导入模型和处理器失败: {e}")
    # 如果导入失败，将在运行时使用回退机制

class MetonymyAwareCaptioner(nn.Module):
    """
    转喻感知字幕生成器
    基于Qwen2-VL生成包含转喻线索的字幕
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct", 
                 device: str = "cuda",
                 local_cache_dir: str = "./models",
                 timeout: int = 3,
                 model_provider: str = "local"):
                 
        super().__init__()
        self.device = device
        self.local_cache_dir = local_cache_dir
        self.timeout = timeout
        self.model_provider = model_provider
        
        # 确保本地缓存目录存在
        import os
        os.makedirs(local_cache_dir, exist_ok=True)
        
        # CPU模式下直接使用回退机制，避免内存不足
        if device == "cpu":
            print("CPU模式内存限制，使用回退机制")
            self.use_fallback = True
        else:
            self.use_fallback = False
            try:
                # 检查本地是否已有模型文件
                import glob
                
                # 构建本地模型路径
                if model_provider == "local":
                    # 直接使用本地模型路径
                    model_path = local_cache_dir
                    print(f"尝试使用本地模型: {model_path}")
                    
                    # 检查模型文件是否存在
                    model_files = glob.glob(os.path.join(model_path, "**", "*.safetensors"), recursive=True)
                    has_local_model = len(model_files) > 0
                    
                    if not has_local_model:
                        print("本地模型文件不存在，使用回退机制")
                        self.use_fallback = True
                        return

                else:

                    # 检查本地缓存是否已有模型文件
                    model_files = glob.glob(os.path.join(local_cache_dir, "**", "*.bin"), recursive=True)
                    has_local_model = len(model_files) > 0
                    
                    # 首先检查网络连接
                    def check_internet_connection():
                        try:
                            import socket
                            if model_provider == "modelscope":
                                socket.create_connection(("modelscope.cn", 443), timeout=timeout)
                            else:
                                socket.create_connection(("huggingface.co", 443), timeout=timeout)
                            return True
                        except (socket.timeout, OSError):
                            return False
                    
                    # 检查网络连接
                    has_internet = check_internet_connection()
                    
                    print(f"尝试从 {model_provider} 加载模型: {model_name}...")
                    print(f"本地缓存目录: {local_cache_dir}")
                    print(f"本地是否已有模型文件: {'是' if has_local_model else '否'}")
                    print(f"网络连接状态: {'已连接' if has_internet else '未连接'}")
                    
                    # 如果没有网络连接且没有本地模型，直接使用回退机制
                    if not has_internet and not has_local_model:
                        print("无网络连接且无本地模型，使用回退机制")
                        self.use_fallback = True
                        return
                    
                    # 从 modelscope 下载模型
                    if model_provider == "modelscope":
                        print("从 modelscope 下载模型...")
                        try:
                            # 设置下载超时
                            import os
                            os.environ['MODEL_SCOPE_DOWNLOAD_TIMEOUT'] = str(timeout * 60)  # 转换为秒
                            
                            model_path = snapshot_download(
                                model_name,
                                cache_dir=local_cache_dir,
                                revision="master"
                            )
                            print(f"模型下载成功，路径: {model_path}")
                        except Exception as e:
                            print(f"模型下载失败: {e}")
                            print("使用默认的简单字幕生成器")
                            self.use_fallback = True
                            return
                    else:
                        model_path = model_name
                
                # 检测模型类型并加载相应的处理器和模型
                print("加载处理器...")
                
                # 尝试从config.json文件中读取模型类型
                import json
                model_type = None
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            model_type = config.get("model_type", None)
                            print(f"从config.json检测到模型类型: {model_type}")
                    except Exception as e:
                        print(f"读取config.json失败: {e}")
                
                try:
                    # GPU模式下使用bfloat16精度
                    dtype = torch.bfloat16
                    print("GPU模式，使用bfloat16精度")
                    
                    if model_type == "qwen2_vl":
                        # 使用 Qwen2-VL 处理器和模型
                        print("使用 Qwen2-VL 处理器和模型...")
                        # 尝试使用 Qwen2-VL 处理器
                        self.processor = Qwen2VLProcessor.from_pretrained(
                            model_path,
                            cache_dir=local_cache_dir,
                            force_download=False
                        )
                        print("使用 Qwen2-VL 处理器")
                        
                        # 尝试使用 Qwen2-VL 模型
                        print("加载模型...")
                        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                            model_path,
                            dtype=dtype,
                            device_map=device,
                            cache_dir=local_cache_dir,
                            force_download=False
                        )
                        print("使用 Qwen2-VL 模型")
                    elif model_type == "qwen3_vl":
                        # 使用 Qwen3-VL 处理器和模型
                        print("使用 Qwen3-VL 处理器和模型...")
                        # 尝试使用 Qwen3-VL 处理器
                        self.processor = Qwen3VLProcessor.from_pretrained(
                            model_path,
                            cache_dir=local_cache_dir,
                            force_download=False
                        )
                        print("使用 Qwen3-VL 处理器")
                        
                        # 尝试使用 Qwen3-VL 模型
                        print("加载模型...")
                        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                            model_path,
                            dtype=dtype,
                            device_map=device,
                            cache_dir=local_cache_dir,
                            force_download=False
                        )
                        print("使用 Qwen3-VL 模型")
                    else:
                        # 模型类型未知，尝试两种模型
                        print("模型类型未知，尝试使用 Qwen2-VL 处理器和模型...")
                        # 尝试使用 Qwen2-VL 处理器
                        self.processor = Qwen2VLProcessor.from_pretrained(
                            model_path,
                            cache_dir=local_cache_dir,
                            force_download=False
                        )
                        print("使用 Qwen2-VL 处理器")
                        
                        # 尝试使用 Qwen2-VL 模型
                        print("加载模型...")
                        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                            model_path,
                            dtype=dtype,
                            device_map=device,
                            cache_dir=local_cache_dir,
                            force_download=False
                        )
                        print("使用 Qwen2-VL 模型")
                except Exception as e:
                    print(f"加载模型失败: {e}")
                    print("使用默认的简单字幕生成器")
                    self.use_fallback = True
                    return
                print("模型加载成功！")
            except Exception as e:
                print(f"模型加载失败: {e}")
                print("使用默认的简单字幕生成器")
                self.use_fallback = True
        
        # 转喻感知的Prompt模板
        self.caption_prompt = """请描述这张图片，并特别注意以下几点：
1. 图片中有哪些物体？
2. 这些物体可能暗示什么样的情感或氛围？
3. 如果这张图片是用户评论的一部分，用户可能想通过图片表达什么？

请生成一个详细的描述，突出可能的情感暗示。"""
        
    def generate_caption(self, image: Image.Image, 
                        review_text: Optional[str] = None,
                        metonymy_type: Optional[str] = None,
                        sentiment: Optional[str] = None) -> str:
        """
        生成转喻增强字幕
        
        Args:
            image: 输入图片
            review_text: 用户评论文本
            metonymy_type: 转喻类型 (T1, T2, T3 或 None)
            sentiment: 情感倾向 (positive, negative, neutral)
        """
        if self.use_fallback:
            # 使用简单的回退方法
            width, height = image.size
            fallback_info = f"图片大小: {width}x{height}, 模式: {image.mode}"
            if metonymy_type:
                fallback_info += f", 转喻类型: {metonymy_type}"
            if sentiment:
                fallback_info += f", 情感: {sentiment}"
            return fallback_info
        
        try:
            # 构建对话
            prompt_parts = []
            
            if review_text:
                prompt_parts.append(f"用户评论：{review_text}")
            
            if metonymy_type:
                metonymy_desc = {
                    "T1": "生理效果代情绪（图片展示行为/表情的结果，暗示用户的情绪）",
                    "T2": "感知物代感知者情绪（图片展示物体，暗示用户对该物体的情感态度）",
                    "T3": "外景代内情（图片展示场景，暗示用户在该场景下的内心感受）"
                }
                prompt_parts.append(f"转喻类型：{metonymy_desc.get(metonymy_type, metonymy_type)}")
            
            if sentiment:
                sentiment_desc = {
                    "positive": "积极/正面",
                    "negative": "消极/负面",
                    "neutral": "中性"
                }
                prompt_parts.append(f"情感倾向：{sentiment_desc.get(sentiment, sentiment)}")
            
            prompt_parts.append(self.caption_prompt)
            
            prompt = "\n".join(prompt_parts)
                
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 处理输入
            print("处理输入...")
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            print(f"输入处理完成，input_ids形状: {inputs['input_ids'].shape}")
            
            # 生成
            print("开始生成字幕...")
            with torch.no_grad():
                # 调整生成参数以适应 Qwen3-VL 模型，移除不支持的参数
                # 减少max_new_tokens以加快生成速度
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )
            print(f"字幕生成完成，generated_ids形状: {generated_ids.shape}")
            
            # 解码
            generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
            caption = self.processor.decode(
                generated_ids[0], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            print(f"字幕解码完成，字幕长度: {len(caption)}")
            
            return caption
        except Exception as e:
            print(f"生成字幕失败: {e}")
            # 使用回退方法
            width, height = image.size
            return f"图片大小: {width}x{height}, 模式: {image.mode}"
    
    def batch_generate(self, images: List[Image.Image], 
                      review_texts: Optional[List[str]] = None) -> List[str]:
        """
        批量生成字幕
        """
        captions = []
        for i, image in enumerate(images):
            text = review_texts[i] if review_texts else None
            cap = self.generate_caption(image, text)
            captions.append(cap)
            
            if (i + 1) % 10 == 0:
                print(f"已生成 {i+1}/{len(images)} 条字幕")
                
        return captions