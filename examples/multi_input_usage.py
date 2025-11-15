"""
Example: Multi-Input Augmentation with Paraphrases and Multi-View Images

This example demonstrates how to use the dynamic multi-input feature
for both text (paraphrases) and images (multi-view) with ensemble encoding.

The model will:
1. Generate N paraphrases of each question
2. Generate M augmented views of each image
3. Encode all paraphrases and aggregate embeddings (mean/sum/max)
4. Encode all image views and aggregate embeddings (mean/sum/max)
5. Combine aggregated features for final prediction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoImageProcessor
from augmentation import AugmentationFactory
from dataset import ViVQADataset
from models import SimpleVQA, SimpleVQAConfig


def example_paraphrase_text_augmentation():
    """Example: Using paraphrase augmentation for ensemble text encoding."""
    print("=" * 60)
    print("Example 1: Paraphrase Text Augmentation")
    print("=" * 60)
    
    # Create paraphrase augmentation (returns list of texts)
    text_aug = AugmentationFactory.create_text_augmentation(
        augmentation_type='simple-paraphrase',
        difficulty=0.5,
        num_paraphrases=3  # Generate 3 paraphrases (+ original = 4 total)
    )
    
    # Test augmentation
    sample_question = "Có bao nhiêu con chó trong hình?"
    paraphrases = text_aug.augment(sample_question)
    
    print(f"\nOriginal: {sample_question}")
    print(f"Generated {len(paraphrases)} paraphrases:")
    for i, para in enumerate(paraphrases, 1):
        print(f"  {i}. {para}")
    
    print("\nNote: These paraphrases will be encoded separately and aggregated")
    print("      using mean/sum/max pooling as configured in the model.")
    

def example_multi_view_image_augmentation():
    """Example: Using multi-view augmentation for ensemble image encoding."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-View Image Augmentation")
    print("=" * 60)
    
    from PIL import Image
    import numpy as np
    
    # Create a dummy image
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    
    # Create multi-view augmentation (returns list of images)
    image_aug = AugmentationFactory.create_image_augmentation(
        augmentation_type='multi-view',
        difficulty=0.5,
        num_views=4  # Generate 4 different views
    )
    
    views = image_aug.augment(dummy_image)
    
    print(f"\nGenerated {len(views)} image views:")
    print(f"  - Original image (included)")
    print(f"  - {len(views)-1} augmented views with rotation, brightness, contrast variations")
    print("\nNote: These views will be encoded separately and aggregated")
    print("      using mean/sum/max pooling as configured in the model.")


def example_full_pipeline_with_multi_input():
    """Example: Full training pipeline with multi-input augmentation."""
    print("\n" + "=" * 60)
    print("Example 3: Full Pipeline with Multi-Input Augmentation")
    print("=" * 60)
    
    # Setup
    dataset_name = 'vivqa'
    train_ann_path = 'data/vivqa/train.csv'
    train_img_dir = 'data/vivqa/images'
    
    # Check if files exist
    if not os.path.exists(train_ann_path):
        print(f"\nSkipping: Dataset not found at {train_ann_path}")
        print("Run this example after downloading the ViVQA dataset.")
        return
    
    # Initialize processors
    tokenizer = AutoTokenizer.from_pretrained('vinai/bartpho-syllable-base')
    processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # Create multi-input augmentations
    text_aug = AugmentationFactory.create_text_augmentation(
        augmentation_type='simple-paraphrase',
        difficulty=0.3,
        num_paraphrases=2  # 2 paraphrases + original = 3 texts
    )
    
    image_aug = AugmentationFactory.create_image_augmentation(
        augmentation_type='multi-view',
        difficulty=0.4,
        num_views=3  # 3 image views
    )
    
    # Create dataset with multi-input augmentations
    print("\nCreating dataset with multi-input augmentations...")
    dataset = ViVQADataset(
        ann_path=train_ann_path,
        img_dir=train_img_dir,
        text_processor=tokenizer,
        vis_processor=processor,
        text_augmentation=text_aug.augment,  # Returns list of texts
        image_augmentation=image_aug.augment,  # Returns list of images
        max_length=64
    )
    
    # Get a sample
    print("\nGetting sample from dataset...")
    sample = dataset[0]
    
    print(f"\nSample shapes:")
    print(f"  Image: {sample['image'].shape}")
    print(f"    Expected: [num_views, C, H, W] = [3, 3, 224, 224]")
    print(f"  Question input_ids: {sample['question_input_ids'].shape}")
    print(f"    Expected: [num_paraphrases, seq_len] = [3, 64]")
    print(f"  Question attention_mask: {sample['question_attention_mask'].shape}")
    print(f"  Label: {sample['label'].shape}")
    
    # Create model with aggregation config
    print("\nCreating model with aggregation configuration...")
    model_config = SimpleVQAConfig(
        vis_model_name='google/vit-base-patch16-224',
        text_model_name='vinai/bartpho-syllable-base',
        num_classes=len(dataset.label_encoder),
        hidden_size=768,
        text_aggregation='mean',  # Average paraphrase embeddings
        vis_aggregation='mean'     # Average multi-view embeddings
    )
    
    model = SimpleVQA(model_config)
    model.eval()
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        # Add batch dimension
        image = sample['image'].unsqueeze(0)  # [1, 3, 3, 224, 224]
        input_ids = sample['question_input_ids'].unsqueeze(0)  # [1, 3, 64]
        attention_mask = sample['question_attention_mask'].unsqueeze(0)
        label = sample['label'].unsqueeze(0)
        
        output = model(
            image=image,
            question_input_ids=input_ids,
            question_attention_mask=attention_mask,
            labels=label
        )
    
    print(f"\nModel output:")
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"    Expected: [batch_size, num_classes] = [1, {len(dataset.label_encoder)}]")
    print(f"  Loss: {output['loss'].item():.4f}")
    
    print("\n✓ Successfully processed multi-input sample!")
    print("\nWhat happened internally:")
    print("  1. Text encoder received [1, 3, 64] and encoded 3 paraphrases")
    print("  2. Text embeddings aggregated using 'mean' → [1, 768]")
    print("  3. Vision encoder received [1, 3, 3, 224, 224] and encoded 3 views")
    print("  4. Vision embeddings aggregated using 'mean' → [1, 768]")
    print("  5. Classifier combined both features → [1, num_classes]")


def example_programmatic_configuration():
    """Example: Programmatic configuration for multi-input training."""
    print("\n" + "=" * 60)
    print("Example 4: Programmatic Training Configuration")
    print("=" * 60)
    
    from training import ExperimentConfig, ModelConfig, DataConfig, AugmentationConfig, TrainingConfig
    
    # Configure experiment with multi-input augmentation
    config = ExperimentConfig(
        experiment_name='vivqa_multi_input_ensemble',
        model=ModelConfig(
            vis_model_name='google/vit-base-patch16-224',
            text_model_name='vinai/bartpho-syllable-base',
            hidden_size=768,
            text_aggregation='mean',  # Aggregate paraphrase embeddings
            vis_aggregation='mean',   # Aggregate multi-view embeddings
        ),
        data=DataConfig(
            dataset_name='vivqa',
            batch_size=32,  # Note: Effective batch size is 32 * 3 texts * 3 views
            seq_len=64,
        ),
        augmentation=AugmentationConfig(
            enable_augmentation=True,
            text_augmentation_type='simple-paraphrase',  # Multi-text augmentation
            image_augmentation_type='multi-view',        # Multi-view augmentation
            enable_curriculum=True,
            curriculum_strategy='linear',
        ),
        training=TrainingConfig(
            epochs=30,
            learning_rate=1e-4,
            output_dir='runs/vivqa_multi_input',
        )
    )
    
    print("\nConfiguration created:")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Text aggregation: {config.model.text_aggregation}")
    print(f"  Vision aggregation: {config.model.vis_aggregation}")
    print(f"  Text augmentation: {config.augmentation.text_augmentation_type}")
    print(f"  Image augmentation: {config.augmentation.image_augmentation_type}")
    
    print("\n# To run training with this config:")
    print("python train.py \\")
    print("  --dataset_name vivqa \\")
    print("  --enable_augmentation \\")
    print("  --text_augmentation_type simple-paraphrase \\")
    print("  --image_augmentation_type multi-view \\")
    print("  --text_aggregation mean \\")
    print("  --vis_aggregation mean \\")
    print("  --enable_curriculum \\")
    print("  --epochs 30")


def example_aggregation_methods():
    """Example: Comparing different aggregation methods."""
    print("\n" + "=" * 60)
    print("Example 5: Aggregation Method Comparison")
    print("=" * 60)
    
    print("\nAvailable aggregation methods:")
    print("  - 'mean': Average embeddings (default, most stable)")
    print("  - 'sum': Sum embeddings (preserves total information)")
    print("  - 'max': Max pooling (keeps strongest features)")
    print("  - 'first': Use only first input (disables ensemble)")
    
    print("\nRecommendations:")
    print("  - Use 'mean' for paraphrases (balanced representation)")
    print("  - Use 'max' for multi-view images (capture best features)")
    print("  - Use 'sum' when input count varies (preserves magnitude)")
    
    print("\nExample configurations:")
    configs = [
        ("Balanced ensemble", "mean", "mean"),
        ("Strong features", "mean", "max"),
        ("Information preserving", "sum", "sum"),
        ("Baseline (no ensemble)", "first", "first"),
    ]
    
    for name, text_agg, vis_agg in configs:
        print(f"\n  {name}:")
        print(f"    text_aggregation='{text_agg}', vis_aggregation='{vis_agg}'")


if __name__ == '__main__':
    print("Multi-Input Augmentation Examples")
    print("=" * 60)
    print("\nThese examples demonstrate the dynamic multi-input feature")
    print("that supports both single and multiple inputs for text and images.\n")
    
    # Run examples
    example_paraphrase_text_augmentation()
    example_multi_view_image_augmentation()
    example_full_pipeline_with_multi_input()
    example_programmatic_configuration()
    example_aggregation_methods()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
