import json
import pandas as pd
import os
from .base import BaseDataset

class CombinedDataset(BaseDataset):
    """
    Dataset class that combines an original dataset with SapAugmented data.
    """
    
    def __init__(self, original_dataset, augmented_json_path=None):
        """
        Initialize Combined dataset by combining original dataset with augmented data.
        
        Args:
            original_dataset: Instance of BaseDataset (ViVQADataset, OpenViVQADataset, ViVQAXDataset)
            augmented_json_path: Direct path to augmented JSON file (optional)
        """
        self.original_dataset = original_dataset
        self.text_processor = original_dataset.text_processor
        self.vis_processor = original_dataset.vis_processor
        self.kwargs = original_dataset.kwargs
        
        self.augmented_json_path = augmented_json_path
        
        # Combine datasets
        self.combined_data = self.load_and_combine_data()
        
        # Use label encoder from original dataset (includes all possible answers)
        self.label_encoder = original_dataset.label_encoder
        
        # Process data into the expected format
        self.data = self.process_combined_data()
    
    def load_and_combine_data(self):
        """Load and combine original and augmented datasets."""
        combined_data = []
        
        # Load original dataset data
        original_data = self.original_dataset.data
        for i in range(len(original_data['questions'])):
            combined_data.append({
                "img_path": original_data['img_paths'][i],
                "question": original_data['questions'][i],
                "answer": original_data['answers'][i],
                "source": "original"
            })
        
        # Load augmented dataset if exists
        if self.augmented_json_path and os.path.exists(self.augmented_json_path):
            print(f"✅ Found augmented dataset: {self.augmented_json_path}")
            with open(self.augmented_json_path, 'r', encoding='utf-8') as f:
                augmented_data = json.load(f)
            
            for item in augmented_data:
                # Use augmented question
                question = item.get('augmented_question', item.get('question', ''))
                combined_data.append({
                    "img_path": item['img_path'],
                    "question": question,
                    "answer": item['answer'],
                    "source": "sap_augmented",
                    "original_question": item.get('original_question', ''),
                    "lambda_value": item.get('lambda_value', 0.5)
                })
        else:
            print(f"⚠️  Augmented dataset not found: {self.augmented_json_path}")
            print(f"   Will use only original dataset for training")
        
        original_count = sum(1 for x in combined_data if x['source'] == 'original')
        augmented_count = sum(1 for x in combined_data if x['source'] == 'sap_augmented')        
        print(f"✅ Combined dataset statistics:")
        print(f"   Original samples: {original_count}")
        print(f"   Augmented samples: {augmented_count}")
        print(f"   Total samples: {len(combined_data)}")
        
        return combined_data
    
    def process_combined_data(self):
        """Process combined data into the format expected by BaseDataset."""
        processed_data = {
            "img_paths": [],
            "questions": [],
            "answers": []
        }
        
        for item in self.combined_data:
            processed_data["img_paths"].append(item['img_path'])
            processed_data["questions"].append(item['question'])
            processed_data["answers"].append(item['answer'])
        
        return processed_data
    
    def get_augmentation_stats(self):
        """Get statistics about the augmentation in the dataset."""
        original_count = sum(1 for item in self.combined_data if item.get('source') == 'original')
        augmented_count = sum(1 for item in self.combined_data if item.get('source') == 'sap_augmented')
        
        return {
            "total_samples": len(self.combined_data),
            "original_samples": original_count,
            "augmented_samples": augmented_count,
            "augmentation_ratio": augmented_count / len(self.combined_data) if len(self.combined_data) > 0 else 0
        }
    
    def get_label_encoder(self):
        return self.original_dataset.get_label_encoder()