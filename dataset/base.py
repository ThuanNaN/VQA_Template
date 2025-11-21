import os
import json
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from typing import Optional, Callable


class BaseDataset(Dataset):
    def __init__(self, ann_path, img_dir, 
                 text_processor, vis_processor,
                 image_augmentation: Optional[Callable] = None,
                 text_augmentation: Optional[Callable] = None,
                 **kwargs):
        super(BaseDataset, self).__init__()
        self.img_dir = img_dir
        self.text_processor = text_processor
        self.vis_processor = vis_processor
        self.image_augmentation = image_augmentation
        self.text_augmentation = text_augmentation
        self.label_encoder = self.get_label_encoder()
        self.data = self.load_data(ann_path)
        self.kwargs = kwargs

    def __len__(self):
        return len(self.data["questions"])

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        All augmentations now return lists for multi-view support:
        - Image augmentation returns list of PIL Images
        - Text augmentation returns list of strings
        - If no augmentation, lists contain only the original
        
        Returns:
            Dict with:
                - image: [C, H, W] or [num_images, C, H, W]
                - question_input_ids: [seq_len] or [num_texts, seq_len]
                - question_attention_mask: [seq_len] or [num_texts, seq_len]
                - label: scalar tensor
        """
        img_path = self.data['img_paths'][idx]
        pil_image = Image.open(img_path).convert('RGB')
        
        # Apply image augmentation if provided
        # Augmentation always returns list of images
        if self.image_augmentation is not None:
            augmented_images = self.image_augmentation(pil_image)
            image = self.vis_processor(augmented_images, return_tensors="pt")["pixel_values"].squeeze(0) 
        else:
            # No augmentation - process single image
            image = self.vis_processor(pil_image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        question_text = self.data['questions'][idx]
        
        # Apply text augmentation if provided
        # Augmentation always returns list of texts
        if self.text_augmentation is not None:
            augmented_texts = self.text_augmentation(question_text)
            
            # Process all texts in the list
            questions = self.text_processor(
                augmented_texts,  # List of strings
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                **self.kwargs
            )
            
            if len(augmented_texts) == 1:
                question_input_ids = questions["input_ids"].squeeze(0)  # [seq_len]
                question_attention_mask = questions["attention_mask"].squeeze(0)  # [seq_len]
            else:
                question_input_ids = questions["input_ids"]  # [num_texts, seq_len]
                question_attention_mask = questions["attention_mask"]  # [num_texts, seq_len]
        else:
            # No augmentation - process single text
            question = self.text_processor(
                question_text, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                **self.kwargs
            )
            question_input_ids = question["input_ids"].squeeze(0)
            question_attention_mask = question["attention_mask"].squeeze(0)

        answer = self.data['answers'][idx]
        answer_label = self.label_encoder.get(answer, None)
        if answer_label is None:
            raise ValueError(f"Label for answer '{answer}' not found in label encoder")
        answer_label = torch.tensor(answer_label, dtype=torch.long)

        return {
            "image": image,
            "question_input_ids": question_input_ids,
            "question_attention_mask": question_attention_mask,
            "label": answer_label
        }

    def load_data(self, ann_path):
        if ann_path.endswith('.csv'):
            data = self.process_csv(ann_path)
        elif ann_path.endswith('.json'):
            data = self.process_json(ann_path)
        else:
            raise ValueError(f"Unsupported file format for {ann_path}")
        return data

    def process_csv(self, ann_path) -> dict:
        data = pd.read_csv(ann_path)
        questions = []
        answers = []
        img_paths = []
        for _, item in data.iterrows():
            questions.append(item['question'])
            answers.append(item['answer'])
            img_id = str(item['img_id']).zfill(12)
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            img_paths.append(img_path)
        return {
            'questions': questions,
            'answers': answers,
            'img_paths': img_paths
        }

    def process_json(self, ann_path) -> dict:
        with open(ann_path, 'r') as f:
            anns = json.load(f)
        image_anns = anns['images']
        iqa_anns = anns['annotations']
        keys_id = list(iqa_anns.keys())
        questions = []
        answers = []
        img_paths = []
        for idx in keys_id:
            iqa = iqa_anns[idx]
            img_id = iqa['image_id']
            img_path = os.path.join(self.img_dir, image_anns[str(img_id)])
            questions.append(iqa['question'])
            answers.append(iqa['answer'])
            img_paths.append(img_path)
        return {
            'questions': questions,
            'answers': answers,
            'img_paths': img_paths
        }

    def get_label_encoder(self):
        raise NotImplementedError
    
    def set_image_augmentation(self, augmentation: Optional[Callable]):
        """
        Set or update the image augmentation function.
        
        Args:
            augmentation: Callable that takes PIL Image and returns list of augmented PIL Images
        """
        self.image_augmentation = augmentation
    
    def set_text_augmentation(self, augmentation: Optional[Callable]):
        """
        Set or update the text augmentation function.
        
        Args:
            augmentation: Callable that takes text string and returns list of augmented text strings
        """
        self.text_augmentation = augmentation