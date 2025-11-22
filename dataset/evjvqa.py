import os
import json
from .base import BaseDataset

class EVJVQADataset(BaseDataset):
    """
    EVJVQA (English-Vietnamese-Japanese VQA) Dataset Loader
    
    Dataset structure:
    - data/evjvqa/evjvqa_train.json
    
    Images should be placed in:
    - data/evjvqa/images/
    
    Note: This dataset contains English questions and answers
    """
    
    train_ann = "data/evjvqa/evjvqa_train.json"
    val_ann = "data/evjvqa/evjvqa_val.json"
    test_ann = "data/evjvqa/evjvqa_test.json"

    def __init__(self, ann_path, img_dir, text_processor, vis_processor, **kwargs):
        """
        Args:
            ann_path: Path to annotation file
            img_dir: Directory containing EVJVQA images
            text_processor: Text processor for questions
            vis_processor: Visual processor for images
            **kwargs: Additional arguments for text processor
        """
        super().__init__(ann_path, img_dir, text_processor, vis_processor, **kwargs)

    def get_label_encoder(self):
        """Build label encoder from all splits (train, val, test)"""
        all_answers = []
        
        # Collect answers from all splits
        for ann_file in [self.train_ann, self.val_ann, self.test_ann]:
            if os.path.exists(ann_file):
                with open(ann_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # EVJVQA annotations is a list
                    for annotation in data['annotations']:
                        if 'answer' in annotation:
                            all_answers.append(annotation['answer'])
        
        # Create sorted label encoder
        sorted_answers = sorted(set(all_answers))
        return {answer: i for i, answer in enumerate(sorted_answers)}

    def process_json(self, ann_path) -> dict:
        """
        Process EVJVQA JSON format
        
        JSON structure:
        {
            "images": [
                {
                    "id": 2301,
                    "filename": "00000002301.jpg"
                },
                ...
            ],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 2301,
                    "question": "what color is the shirt of the girl wearing glasses?",
                    "answer": "the girl wearing glasses wears a red shirt"
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
            answers.append(annotation['answer'])
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
