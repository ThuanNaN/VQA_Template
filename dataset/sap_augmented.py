import json
import pandas as pd
import os
from .base import BaseDataset

class CombinedDataset(BaseDataset):
    """
    Dataset class that combines an original dataset with SapAugmented data.
    """
    
    def __init__(self, original_dataset, augmented_json_path=None, augmented_dir="augmented_datasets", dataset_name=None):
        """
        Initialize Combined dataset by combining original dataset with augmented data.
        
        Args:
            original_dataset: Instance of BaseDataset (ViVQADataset, OpenViVQADataset, ViVQAXDataset)
            augmented_json_path: Direct path to augmented JSON file (optional)
            augmented_dir: Directory containing augmented datasets (default: "augmented_datasets")
            dataset_name: Dataset name for auto-detecting augmented file ('ViVQA', 'OpenViVQA', 'ViVQA-X')
        """
        self.original_dataset = original_dataset
        self.text_processor = original_dataset.text_processor
        self.vis_processor = original_dataset.vis_processor
        self.kwargs = original_dataset.kwargs
        
        # Auto-detect augmented file path if not provided
        if augmented_json_path is None and dataset_name is not None:
            augmented_json_path = os.path.join(augmented_dir, f"{dataset_name}_sap_augmented.json")
        
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
        
        print(f"🔄 Combined dataset loaded:")
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

class CombinedDatasetFromFile(BaseDataset):
    """
    Dataset class for loading pre-combined dataset from a single JSON file.
    """
    
    def __init__(self, ann_path, img_dir, text_processor, vis_processor, **kwargs):
        """
        Initialize Combined dataset from pre-combined file.
        
        Args:
            ann_path: Path to the pre-combined JSON file
            img_dir: Directory containing images  
            text_processor: Text tokenizer
            vis_processor: Vision processor
        """
        self.img_dir = img_dir
        self.text_processor = text_processor
        self.vis_processor = vis_processor
        self.kwargs = kwargs
        
        # Load combined data
        self.combined_data = self.load_combined_data(ann_path)
        
        # Get label encoder from the combined data
        self.label_encoder = self.get_label_encoder()
        
        # Process data into the expected format
        self.data = self.process_data()
    
    def load_combined_data(self, ann_path):
        """Load the pre-combined JSON file."""
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def get_label_encoder(self):
        """Create label encoder from all answers in the dataset."""
        all_answers = [item['answer'] for item in self.combined_data]
        unique_answers = sorted(set(all_answers))
        return {answer: i for i, answer in enumerate(unique_answers)}
    
    def process_data(self):
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
