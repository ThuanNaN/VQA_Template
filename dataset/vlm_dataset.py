"""
VLM Dataset for VQA task.
Formats data as conversation for chat-based VLMs like Qwen2-VL.
"""

import os
import json
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class VLMVQADataset(Dataset):
    """
    Dataset for VLM fine-tuning on VQA task.
    Formats data as conversation for chat-based VLMs.
    
    Args:
        ann_path: Path to annotation file (csv or json)
        img_dir: Directory containing images
        label_encoder: Dictionary mapping answers to indices
        split: Dataset split name (train/val/test)
        system_prompt: Custom system prompt (optional)
    """
    
    DEFAULT_SYSTEM_PROMPT = """Bạn là một mô hình AI chuyên phân tích hình ảnh và trả lời câu hỏi.
Hãy trả lời câu hỏi một cách ngắn gọn, chính xác dựa trên hình ảnh được cung cấp.
Chỉ trả lời đúng nội dung được hỏi, không giải thích thêm."""

    DEFAULT_SYSTEM_PROMPT_EN = """You are an AI model specialized in analyzing images and answering questions.
Answer questions concisely and accurately based on the provided image.
Only answer what is asked, do not explain further."""
    
    def __init__(
        self,
        ann_path: str,
        img_dir: str,
        label_encoder: Dict[str, int],
        split: str = "train",
        system_prompt: Optional[str] = None,
        language: str = "vi"
    ):
        self.img_dir = img_dir
        self.label_encoder = label_encoder
        self.label_decoder = {v: k for k, v in label_encoder.items()}
        self.split = split
        self.language = language
        
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = (
                self.DEFAULT_SYSTEM_PROMPT if language == "vi" 
                else self.DEFAULT_SYSTEM_PROMPT_EN
            )
        
        self.data = self._load_data(ann_path)
        
    def _load_data(self, ann_path: str) -> List[Dict]:
        """Load and process annotation file"""
        data = []
        
        if ann_path.endswith('.csv'):
            data = self._load_csv(ann_path)
        elif ann_path.endswith('.json'):
            data = self._load_json(ann_path)
        else:
            raise ValueError(f"Unsupported file format: {ann_path}")
        
        return data
    
    def _load_csv(self, ann_path: str) -> List[Dict]:
        """Load CSV annotation file (ViVQA format)"""
        data = []
        df = pd.read_csv(ann_path)
        
        for _, row in df.iterrows():
            img_id = str(row['img_id']).zfill(12)
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            data.append({
                'question': row['question'],
                'answer': str(row['answer']),
                'img_path': img_path
            })
        
        return data
    
    def _load_json(self, ann_path: str) -> List[Dict]:
        """Load JSON annotation file (OpenViVQA, ViVQA-X format)"""
        data = []
        
        with open(ann_path, 'r', encoding='utf-8') as f:
            anns = json.load(f)
        
        # OpenViVQA format: {'images': {...}, 'annotations': {...}}
        if 'annotations' in anns and 'images' in anns:
            image_anns = anns['images']
            iqa_anns = anns['annotations']
            
            for idx, iqa in iqa_anns.items():
                img_id = iqa['image_id']
                img_path = os.path.join(self.img_dir, image_anns[str(img_id)])
                data.append({
                    'question': iqa['question'],
                    'answer': str(iqa['answer']),
                    'img_path': img_path
                })
        
        # ViVQA-X format: list of dicts
        elif isinstance(anns, list):
            for item in anns:
                img_path = os.path.join(self.img_dir, item['image'])
                data.append({
                    'question': item['question'],
                    'answer': str(item['answer']),
                    'img_path': img_path
                })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> List[Dict]:
        """
        Return formatted conversation for VLM.
        
        Returns:
            List of message dicts in conversation format:
            [
                {"role": "system", "content": [...]},
                {"role": "user", "content": [...]},
                {"role": "assistant", "content": [...]}
            ]
        """
        item = self.data[idx]
        
        # Load image
        image = Image.open(item['img_path']).convert('RGB')
        
        # Format as conversation
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": item['question']},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": item['answer']}],
            },
        ]
        
        return conversation
    
    def get_raw_item(self, idx: int) -> Dict:
        """Get raw data item (without conversation formatting)"""
        return self.data[idx]
    
    def get_image(self, idx: int) -> Image.Image:
        """Get PIL Image for an item"""
        return Image.open(self.data[idx]['img_path']).convert('RGB')
    
    def get_question(self, idx: int) -> str:
        """Get question text for an item"""
        return self.data[idx]['question']
    
    def get_answer(self, idx: int) -> str:
        """Get answer text for an item"""
        return self.data[idx]['answer']


class VLMVQADatasetWithAugmentation(VLMVQADataset):
    """
    VLM Dataset with support for augmented data.
    Can combine original and augmented samples.
    """
    
    def __init__(
        self,
        ann_path: str,
        img_dir: str,
        label_encoder: Dict[str, int],
        augmented_json_path: Optional[str] = None,
        max_augmented_samples: Optional[int] = None,
        **kwargs
    ):
        super().__init__(ann_path, img_dir, label_encoder, **kwargs)
        
        self.original_data = self.data.copy()
        self.augmented_data = []
        
        if augmented_json_path and os.path.exists(augmented_json_path):
            self._load_augmented_data(augmented_json_path, max_augmented_samples)
    
    def _load_augmented_data(
        self, 
        augmented_json_path: str, 
        max_samples: Optional[int] = None
    ):
        """Load augmented data from JSON file"""
        with open(augmented_json_path, 'r', encoding='utf-8') as f:
            aug_data = json.load(f)
        
        # Limit samples if specified
        if max_samples is not None:
            aug_data = aug_data[:max_samples]
        
        for item in aug_data:
            # Validate answer exists in label encoder
            answer = str(item.get('answer', '')).strip()
            if answer.lower() in {k.lower() for k in self.label_encoder.keys()}:
                self.augmented_data.append({
                    'question': item.get('augmented_question', item.get('question', '')),
                    'answer': answer,
                    'img_path': item['img_path'],
                    'source': 'augmented'
                })
        
        # Mark original data
        for item in self.original_data:
            item['source'] = 'original'
        
        # Combine
        self.data = self.original_data + self.augmented_data
        
        print(f"✅ Loaded {len(self.augmented_data)} augmented samples")
        print(f"✅ Total samples: {len(self.data)} (original: {len(self.original_data)}, augmented: {len(self.augmented_data)})")
