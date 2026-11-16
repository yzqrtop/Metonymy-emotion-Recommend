import base64
import requests
import json
import time
import hashlib
import os
# ========== 强制关闭代理（解决你之前的报错）==========
os.environ["NO_PROXY"] = "api.siliconflow.cn"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image
import io
import concurrent.futures
from functools import lru_cache
import warnings

# 🔥 关闭所有不安全HTTPS警告（彻底不显示）
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # 只显示错误，不显示烦人的信息

class MetonymyAnnotator:
    """
    基于Qwen3-VL的转喻自动标注器（使用SiliconFlow的OpenAI兼容API）
    """
    
    def __init__(self, api_key: str, model: str =  "Qwen/Qwen3-VL-8B-Instruct", 
                 timeout: int = 30, max_retries: int = 2, 
                 cache_dir: str = "./cache", 
                 max_workers: int = 3):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # ✅ 彻底解决代理/SSL错误
        self.session = requests.Session()
        self.session.trust_env = False          # 不使用系统代理
        self.session.verify = False             # 关闭SSL验证
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=2
        )
        self.session.mount("https://", adapter)
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
    def get_image_hash(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def encode_image(self, image_path: str, max_size: int = 1024) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            if max(image.width, image.height) > max_size:
                ratio = max_size / max(image.width, image.height)
                new_size = (int(image.width*ratio), int(image.height*ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=80)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"图片编码失败: {e}")
            return ""
    
    def check_api_health(self) -> bool:
        try:
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5
            }
            response = self.session.post(
                self.base_url,
                json=test_payload,
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def build_prompt(self, review_text: str, rating: Optional[float] = None) -> str:
        few_shot_examples = """
        示例1：
        用户评论："这杯咖啡绝了！"
        配图：[一杯只剩泡沫的空咖啡杯，旁边是吃了一半的曲奇]
        {
            "contains_metonymy": true,
            "metonymy_type": "T1",
            "metonymy_rationale": "空杯暗示用户已经喝完并感到满足",
            "inferred_sentiment": "positive",
            "key_objects": ["空咖啡杯", "吃了一半的曲奇"],
            "confidence": 0.95
        }
        """
        
        prompt = f"""
        你是一个多模态情感分析专家。请判断给定的"用户评论+配图"是否包含"情感转喻"。
        
        【任务定义】
        情感转喻是指：用户不直接表达情感，而是通过图片中的物体、场景、动作结果来暗示自己的情感态度。
        我们关注三类转喻：
        - T1（生理效果代情绪）
        - T2（感知物代感知者情绪）
        - T3（外景代内情）
        
        【参考示例】
        {few_shot_examples}
        
        【当前任务】
         1. 根据"{review_text}"生成三种情感的用户评论；
        
         2. 分析图片，严格按JSON格式输出，不要多余内容：
        {{
            "metonymy_comment": "三种情感转喻评论",
            "contains_metonymy": true/false,
            "metonymy_type": "T1/T2/T3/null",
            "metonymy_rationale": "解释",
            "inferred_sentiment": "positive/negative/neutral",
            "key_objects": [],
            "confidence": 0.0-1.0
        }}
        """
        return prompt
    
    def annotate(self, review_text: str, image_path: str, rating: Optional[float] = None) -> Dict:
        # 加上前置路径
        if not image_path.startswith('./data/Yelp/Yelp_photos~/'):
            image_path = os.path.join('./data/Yelp/Yelp_photos~/', image_path)
        
        # 图片编码
        base64_image = self.encode_image(image_path)
        if not base64_image:
            return {"contains_metonymy":False,"metonymy_type":None,"metonymy_rationale":"图片读取失败","inferred_sentiment":"neutral","key_objects":[],"confidence":0.0}
        
        # 请求体 - 完全按照 test_metonymy_annotation.py 的格式
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.build_prompt(review_text, rating)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "stream": False,
            "max_tokens": 4096
        }

        # 发送请求 - 完全按照 test_metonymy_annotation.py 的格式
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                data=json.dumps(payload),
                proxies={"http": None, "https": None}  # 彻底关闭代理
            )
            response.raise_for_status()
            result = response.json()

            # 输出结果
            print("=" * 50)
            print("模型回复：")
            print(result["choices"][0]["message"]["content"])
            print("=" * 50)

            # 解析结果
            content = result['choices'][0]['message']['content'].strip()
            content = content.replace("```json", "").replace("```", "").strip()
            annotation = json.loads(content)

            return content

        except Exception as e:
            print(f"请求失败：{str(e)}")
            if 'response' in locals():
                print("错误返回：", response.text)
            # 最终失败
            return {
                "contains_metonymy": False,
                "metonymy_type": None,
                "metonymy_rationale": "API请求失败",
                "inferred_sentiment": "neutral",
                "key_objects": [],
                "confidence": 0.0
            }
    
    def batch_annotate(self, df: pd.DataFrame, text_col: str, image_col: str, 
                       rating_col: Optional[str] = None, sample_ratio: float = 0.1) -> pd.DataFrame:
        sample_df = df.sample(frac=sample_ratio, random_state=42)
        tasks = [
            (row[text_col], row[image_col], row[rating_col] if rating_col else None)
            for _, row in sample_df.iterrows()
        ]
        
        annotations = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {executor.submit(self.annotate, *t): i for i, t in enumerate(tasks)}
            for future in concurrent.futures.as_completed(future_to_idx):
                try:
                    annotations.append((future_to_idx[future], future.result()))
                except:
                    annotations.append((future_to_idx[future], {
                        "contains_metonymy":False,"metonymy_type":None,"metonymy_rationale":"批量标注失败",
                        "inferred_sentiment":"neutral","key_objects":[],"confidence":0.0
                    }))
        
        annotations.sort()
        anns = [a[1] for a in annotations]
        for k in ['contains_metonymy','metonymy_type','metonymy_rationale','inferred_sentiment','key_objects','confidence']:
            sample_df[f'metonymy_{k}'] = [a[k] for a in anns]
        return sample_df
    
    def shutdown(self):
        self.executor.shutdown(wait=True)