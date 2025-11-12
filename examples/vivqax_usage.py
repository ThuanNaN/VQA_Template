"""
Example usage of ViVQA-X Dataset

This script demonstrates how to use the ViVQA-X dataset loader.
"""

import sys
sys.path.append('..')

from dataset import ViVQAXDataset
from transformers import AutoTokenizer, AutoImageProcessor
import torch

def main():
    # Initialize processors
    model_name = "vinai/phobert-base"  # Vietnamese language model
    vision_model = "google/vit-base-patch16-224"
    
    text_processor = AutoTokenizer.from_pretrained(model_name)
    vis_processor = AutoImageProcessor.from_pretrained(vision_model)
    
    # Create dataset instances
    print("Loading ViVQA-X dataset...")
    
    # Training set
    train_dataset = ViVQAXDataset(
        ann_path="data/vivqax/train.json",
        img_dir="data/MSCOCO",  # COCO images directory
        text_processor=text_processor,
        vis_processor=vis_processor,
        include_explanations=True,  # Include explanations in output
        max_length=128
    )
    
    # Validation set
    val_dataset = ViVQAXDataset(
        ann_path="data/vivqax/val.json",
        img_dir="data/MSCOCO",
        text_processor=text_processor,
        vis_processor=vis_processor,
        include_explanations=True,
        max_length=128
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of answer classes: {len(train_dataset.label_encoder)}")
    
    # Get a sample
    print("\n" + "="*50)
    print("Sample from dataset:")
    print("="*50)
    sample = train_dataset[0]
    
    print(f"Question ID: {sample['question_id']}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Question input IDs shape: {sample['question_input_ids'].shape}")
    print(f"Answer label: {sample['label']}")
    
    if 'explanation' in sample:
        print(f"Explanation: {sample['explanation']}")
    
    # Decode the question
    question_text = text_processor.decode(sample['question_input_ids'], skip_special_tokens=True)
    print(f"Decoded question: {question_text}")
    
    # Get answer text from label
    label_to_answer = {v: k for k, v in train_dataset.label_encoder.items()}
    answer_text = label_to_answer[sample['label'].item()]
    print(f"Answer: {answer_text}")
    
    # Create DataLoader
    print("\n" + "="*50)
    print("Creating DataLoader:")
    print("="*50)
    
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch question input IDs shape: {batch['question_input_ids'].shape}")
    print(f"Batch labels shape: {batch['label'].shape}")
    
    print("\nDataset loaded successfully!")

if __name__ == "__main__":
    main()
