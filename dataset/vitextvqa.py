import os
import json
from .base import BaseDataset

class ViTextVQADataset(BaseDataset):
    """
    ViTextVQA Dataset Loader
    
    Dataset structure:
    - data/vitextvqa/ViTextVQA_train.json
    - data/vitextvqa/ViTextVQA_dev.json
    - data/vitextvqa/ViTextVQA_test.json
    
    Images should be placed in:
    - data/vitextvqa/images/ (or a custom directory)
    """
    
    train_ann = "data/vitextvqa/ViTextVQA_train.json"
    dev_ann = "data/vitextvqa/ViTextVQA_dev.json"
    test_ann = "data/vitextvqa/ViTextVQA_test.json"

    def __init__(self, ann_path, img_dir, text_processor, vis_processor, **kwargs):
        """
        Args:
            ann_path: Path to annotation file
            img_dir: Directory containing ViTextVQA images
            text_processor: Text processor for questions
            vis_processor: Visual processor for images
            **kwargs: Additional arguments for text processor
        """
        super().__init__(ann_path, img_dir, text_processor, vis_processor, **kwargs)

    def get_label_encoder(self):
        """Build label encoder from all splits"""
        all_answers = []
        
        # Collect answers from all available splits
        for ann_file in [self.train_ann, self.dev_ann, self.test_ann]:
            if os.path.exists(ann_file):
                with open(ann_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # ViTextVQA has multiple answers per question
                    for annotation in data['annotations']:
                        # Take the first answer as the primary answer
                        if annotation['answers']:
                            all_answers.append(annotation['answers'][0])
        
        # Create sorted label encoder
        sorted_answers = sorted(set(all_answers))
        return {answer: i for i, answer in enumerate(sorted_answers)}

    def process_json(self, ann_path) -> dict:
        """
        Process ViTextVQA JSON format
        
        JSON structure:
        {
            "images": [
                {
                    "id": 9067,
                    "filename": "9067.jpg"
                },
                ...
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 0,
                    "question": "quán ăn này bán những món gì ?",
                    "answers": ["mì quảng , bún bò huế"]
                },
                ...
            ]
        }
        """
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create image_id to filename mapping
        image_mapping = {img['id']: img['filename'] for img in data['images']}
        
        questions = []
        answers = []
        img_paths = []
        question_ids = []
        
        for annotation in data['annotations']:
            questions.append(annotation['question'])
            
            # Use the first answer as the primary answer
            # ViTextVQA can have multiple valid answers
            if annotation['answers']:
                answers.append(annotation['answers'][0])
            else:
                # Skip annotations without answers
                continue
            
            question_ids.append(annotation['id'])
            
            # Get image filename from mapping
            image_id = annotation['image_id']
            image_filename = image_mapping.get(image_id, f"{image_id}.jpg")
            
            # Construct full image path
            img_path = os.path.join(self.img_dir, image_filename)
            img_paths.append(img_path)
        
        return {
            'questions': questions,
            'answers': answers,
            'img_paths': img_paths,
            'question_ids': question_ids
        }

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        item = super().__getitem__(idx)
        
        # Add question_id to the output
        item['question_id'] = self.data['question_ids'][idx]
        
        return item
