"""
Example usage of ViVQA Dataset

This script demonstrates how to use the ViVQA dataset loader.
"""

import sys
sys.path.append('..')

from dataset import ViVQADataset
from transformers import AutoTokenizer, AutoImageProcessor
import torch

def main():
    # Initialize processors
    model_name = "vinai/phobert-base"  # Vietnamese language model
    vision_model = "google/vit-base-patch16-224"
    
    text_processor = AutoTokenizer.from_pretrained(model_name)
    vis_processor = AutoImageProcessor.from_pretrained(vision_model)
    
    # Create dataset instances
    print("Loading ViVQA dataset...")
    
    # Training set
    train_dataset = ViVQADataset(
        ann_path="data/vivqa/train.csv",
        img_dir="data/vivqa/images",  # ViVQA images directory
        text_processor=text_processor,
        vis_processor=vis_processor,
        max_length=128
    )
    
    # Test set
    test_dataset = ViVQADataset(
        ann_path="data/vivqa/test.csv",
        img_dir="data/vivqa/images",
        text_processor=text_processor,
        vis_processor=vis_processor,
        max_length=128
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of answer classes: {len(train_dataset.label_encoder)}")
    
    # Get a sample
    print("\n" + "="*50)
    print("Sample from dataset:")
    print("="*50)
    sample = train_dataset[0]
    
    print(f"Image shape: {sample['image'].shape}")
    print(f"Question input IDs shape: {sample['question_input_ids'].shape}")
    print(f"Question attention mask shape: {sample['question_attention_mask'].shape}")
    print(f"Answer label: {sample['label']}")
    
    # Decode the question
    question_text = text_processor.decode(sample['question_input_ids'], skip_special_tokens=True)
    print(f"Decoded question: {question_text}")
    
    # Get answer text from label
    label_to_answer = {v: k for k, v in train_dataset.label_encoder.items()}
    answer_text = label_to_answer[sample['label'].item()]
    print(f"Answer: {answer_text}")
    
    # Print some statistics about the dataset
    print("\n" + "="*50)
    print("Dataset Statistics:")
    print("="*50)
    print(f"Total unique answers: {len(train_dataset.label_encoder)}")
    
    # Show a few examples of answer labels
    print("\nSample answer labels (first 10):")
    for i, (answer, label) in enumerate(list(train_dataset.label_encoder.items())[:10]):
        print(f"  {label}: {answer}")
    
    # Analyze question types (if available in CSV)
    print("\n" + "="*50)
    print("Exploring raw data:")
    print("="*50)
    import pandas as pd
    train_df = pd.read_csv("data/vivqa/train.csv")
    
    if 'type' in train_df.columns:
        print("\nQuestion types distribution:")
        print(train_df['type'].value_counts())
    
    print(f"\nSample questions:")
    for idx in range(min(5, len(train_df))):
        print(f"  Q: {train_df.iloc[idx]['question']}")
        print(f"  A: {train_df.iloc[idx]['answer']}")
        print(f"  Image ID: {train_df.iloc[idx]['img_id']}")
        if 'type' in train_df.columns:
            print(f"  Type: {train_df.iloc[idx]['type']}")
        print()
    
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
    print(f"Batch question attention mask shape: {batch['question_attention_mask'].shape}")
    print(f"Batch labels shape: {batch['label'].shape}")
    
    # Decode questions in the batch
    print("\n" + "="*50)
    print("Sample batch questions and answers:")
    print("="*50)
    for i in range(min(4, batch['question_input_ids'].shape[0])):
        q_text = text_processor.decode(batch['question_input_ids'][i], skip_special_tokens=True)
        a_text = label_to_answer[batch['label'][i].item()]
        print(f"\nSample {i+1}:")
        print(f"  Question: {q_text}")
        print(f"  Answer: {a_text}")
    
    print("\nDataset loaded successfully!")

if __name__ == "__main__":
    main()
