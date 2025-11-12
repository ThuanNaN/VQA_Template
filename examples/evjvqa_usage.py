"""
Example usage of EVJVQA Dataset

This script demonstrates how to use the EVJVQA (English-Vietnamese-Japanese VQA) dataset loader.
EVJVQA is a multilingual Visual Question Answering dataset with English questions and answers.
"""

import sys
sys.path.append('..')

from dataset import EVJVQADataset
from transformers import AutoTokenizer, AutoImageProcessor
import torch

def main():
    # Initialize processors
    # Note: For English text, use an English language model
    model_name = "bert-base-uncased"  # English language model
    vision_model = "google/vit-base-patch16-224"
    
    text_processor = AutoTokenizer.from_pretrained(model_name)
    vis_processor = AutoImageProcessor.from_pretrained(vision_model)
    
    # Create dataset instance
    print("Loading EVJVQA dataset...")
    
    # Training set
    train_dataset = EVJVQADataset(
        ann_path="data/evjvqa/evjvqa_train.json",
        img_dir="data/evjvqa/images",  # EVJVQA images directory
        text_processor=text_processor,
        vis_processor=vis_processor,
        max_length=128
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Number of answer classes: {len(train_dataset.label_encoder)}")
    
    # Get a sample
    print("\n" + "="*50)
    print("Sample from dataset:")
    print("="*50)
    sample = train_dataset[0]
    
    print(f"Question ID: {sample['question_id']}")
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
    
    # Explore the raw data structure
    print("\n" + "="*50)
    print("Exploring raw data structure:")
    print("="*50)
    import json
    with open("data/evjvqa/evjvqa_train.json", 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Number of images: {len(raw_data['images'])}")
    print(f"Number of annotations: {len(raw_data['annotations'])}")
    
    # Show sample annotations
    print("\nSample annotations (first 3):")
    for i, annotation in enumerate(raw_data['annotations'][:3]):
        print(f"\nAnnotation {i+1}:")
        print(f"  Question: {annotation['question']}")
        print(f"  Answer: {annotation['answer']}")
        print(f"  Image ID: {annotation['image_id']}")
        
        # Find the corresponding image
        image_info = next((img for img in raw_data['images'] if img['id'] == annotation['image_id']), None)
        if image_info:
            print(f"  Image filename: {image_info['filename']}")
    
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
        print(f"  Question ID: {batch['question_id'][i]}")
        print(f"  Question: {q_text}")
        print(f"  Answer: {a_text}")
    
    print("\nDataset loaded successfully!")
    print("\nNote: EVJVQA is a multilingual dataset with English questions and answers.")
    print("The dataset name stands for English-Vietnamese-Japanese VQA.")

if __name__ == "__main__":
    main()
