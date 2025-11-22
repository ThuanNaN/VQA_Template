import os
import json
from .base import BaseDataset

class ViVQAXDataset(BaseDataset):
    """
    ViVQA-X Dataset Loader
    
    Dataset structure:
    - data/vivqax/ViVQA-X_train.json
    - data/vivqax/ViVQA-X_val.json
    - data/vivqax/ViVQA-X_test.json
    
    Images are from MS COCO dataset and should be placed in:
    - data/MSCOCO/train2014/
    - data/MSCOCO/val2014/
    """
    
    train_ann = "data/vivqax/ViVQA-X_train.json"
    val_ann = "data/vivqax/ViVQA-X_val.json"
    test_ann = "data/vivqax/ViVQA-X_test.json"

    def __init__(self, ann_path, img_dir, text_processor, vis_processor, 
                 include_explanations=False, **kwargs):
        """
        Args:
            ann_path: Path to annotation file
            img_dir: Directory containing COCO images
            text_processor: Text processor for questions
            vis_processor: Visual processor for images
            include_explanations: Whether to include explanations in the output
            **kwargs: Additional arguments for text processor
        """
        self.include_explanations = include_explanations
        super().__init__(ann_path, img_dir, text_processor, vis_processor, **kwargs)

    def get_label_encoder(self):
        """Build label encoder from all splits"""
        all_answers = []
        
        # Collect answers from all available splits
        for ann_file in [self.train_ann, self.val_ann, self.test_ann]:
            if os.path.exists(ann_file):
                with open(ann_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_answers.extend([item["answer"] for item in data])
        
        # Create sorted label encoder
        sorted_answers = sorted(set(all_answers))
        return {answer: i for i, answer in enumerate(sorted_answers)}

    def process_json(self, ann_path) -> dict:
        """
        Process ViVQA-X JSON format
        
        JSON structure:
        [
            {
                "question": "Đây là phòng nào?",
                "image_id": "524822",
                "image_name": "COCO_val2014_000000524822.jpg",
                "explanation": ["explanation1", "explanation2"],
                "answer": "phòng khách",
                "question_id": "524822007"
            },
            ...
        ]
        """
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        answers = []
        img_paths = []
        explanations = [] if self.include_explanations else None
        question_ids = []
        
        for item in data:
            questions.append(item['question'])
            answers.append(item['answer'])
            question_ids.append(item['question_id'])
            
            # Build image path from image_name
            # image_name format: COCO_train2014_000000262146.jpg or COCO_val2014_000000524822.jpg
            image_name = item['image_name']
            
            # The img_dir already points to the correct split folder (e.g., data/MSCOCO/train2014)
            # So we just need to join img_dir with the image_name
            img_path = os.path.join(self.img_dir, image_name)
            
            img_paths.append(img_path)
            
            if self.include_explanations:
                # Join multiple explanations with separator
                explanation_text = " | ".join(item['explanation'])
                explanations.append(explanation_text)
        
        result = {
            'questions': questions,
            'answers': answers,
            'img_paths': img_paths,
            'question_ids': question_ids
        }
        
        if self.include_explanations:
            result['explanations'] = explanations
        
        return result

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        item = super().__getitem__(idx)
        
        # Add question_id to the output
        item['question_id'] = self.data['question_ids'][idx]
        
        # Add explanation if requested
        if self.include_explanations and 'explanations' in self.data:
            item['explanation'] = self.data['explanations'][idx]
        
        return item


class ViVQAXAddonDataset(ViVQAXDataset):
    """ViVQA-X dataset with addon training data"""
    train_ann = "data/vivqax/ViVQA-X_train_addon.json"
    val_ann = "data/vivqax/ViVQA-X_val.json"
    test_ann = "data/vivqax/ViVQA-X_test.json"
