"""
Example: Training with Curriculum Learning and Augmentation

This example demonstrates how to use the new OOP architecture for training
VQA models with curriculum learning and augmentation.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training import (
    ExperimentConfig,
    ModelConfig,
    DataConfig,
    AugmentationConfig,
    TrainingConfig,
    VQATrainingPipeline,
)
from augmentation import (
    AugmentationFactory,
    MaskedImageAugmentation,
    CurriculumLearningScheduler,
    DifficultyLevel,
)
from PIL import Image
import numpy as np


def example_1_basic_augmentation():
    """Example 1: Basic image augmentation without curriculum learning."""
    print("=" * 80)
    print("Example 1: Basic Image Augmentation")
    print("=" * 80)
    
    # Create an augmentor using factory
    augmentor = AugmentationFactory.create_image_augmentation(
        augmentation_type='masked',
        difficulty=DifficultyLevel.MEDIUM,
        patch_size=16,
        seed=42
    )
    
    # Create a sample image
    sample_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    
    # Apply augmentation
    augmented_image = augmentor.augment(sample_image)
    
    # Get augmentation info
    info = augmentor.get_augmentation_info()
    print(f"Augmentation Info: {info}")
    
    # Change difficulty
    augmentor.set_difficulty(DifficultyLevel.HARD)
    harder_augmented = augmentor.augment(sample_image)
    
    print(f"New difficulty: {augmentor.difficulty.value}")
    print()


def example_2_curriculum_learning():
    """Example 2: Curriculum learning scheduler."""
    print("=" * 80)
    print("Example 2: Curriculum Learning Scheduler")
    print("=" * 80)
    
    # Create curriculum scheduler
    scheduler = CurriculumLearningScheduler(
        total_epochs=30,
        easy_epochs=9,
        medium_epochs=9,
        hard_epochs=12
    )
    
    # Print schedule info
    schedule_info = scheduler.get_schedule_info()
    print("Curriculum Schedule:")
    for line in schedule_info['schedule']:
        print(f"  {line}")
    
    # Simulate training loop
    print("\nEpoch progression:")
    for epoch in range(0, 30, 5):
        difficulty = scheduler.get_difficulty_for_epoch(epoch)
        print(f"  Epoch {epoch:2d}: {difficulty.value.upper()}")
    
    print()


def example_3_programmatic_training():
    """Example 3: Programmatic training configuration."""
    print("=" * 80)
    print("Example 3: Programmatic Training Configuration")
    print("=" * 80)
    
    # Create configuration programmatically
    config = ExperimentConfig(
        model=ModelConfig(
            vis_model_name='google/vit-base-patch16-224',
            text_model_name='vinai/bartpho-syllable-base'
        ),
        data=DataConfig(
            dataset_name='vivqa',
            batch_size=32,
            seq_len=64,
            dataloader_workers=4
        ),
        augmentation=AugmentationConfig(
            enable_augmentation=True,
            augmentation_type='masked',
            patch_size=16,
            seed=42,
            enable_curriculum=True,
            total_epochs=30,
            easy_epochs=9,
            medium_epochs=9,
            hard_epochs=12
        ),
        training=TrainingConfig(
            seed=42,
            epochs=30,
            learning_rate=5e-5,
            weight_decay=1e-4,
            gradient_accumulation=2,
            warmup_steps=500,
            fp16=True,
            patience=5,
            logging_steps=100,
            output_dir='runs/experiment_1',
            run_name='curriculum-v1',
            report_to_wandb=False,  # Set to True to enable WandB
            wandb_project='VQA-Experiments',
            n_threads=8
        )
    )
    
    # Print configuration
    print("Experiment Configuration:")
    config_dict = config.to_dict()
    for section, values in config_dict.items():
        print(f"\n{section.upper()}:")
        for key, value in values.items():
            print(f"  {key}: {value}")
    
    # Create and run pipeline (commented out - requires data)
    # pipeline = VQATrainingPipeline(config)
    # pipeline.run()
    
    print("\nConfiguration created successfully!")
    print("To run training, uncomment the pipeline lines above.")
    print()


def example_4_custom_augmentation():
    """Example 4: Creating custom augmentation strategy."""
    print("=" * 80)
    print("Example 4: Custom Augmentation Strategy")
    print("=" * 80)
    
    from augmentation.base import BaseImageAugmentation
    from PIL import ImageFilter
    import random
    
    class CustomBlurAugmentation(BaseImageAugmentation):
        """Custom blur augmentation with curriculum learning support."""
        
        def _configure_parameters(self):
            """Configure blur parameters based on difficulty."""
            if self.difficulty == DifficultyLevel.EASY:
                self.blur_range = (0.1, 0.5)
                self.probability = 0.3
            elif self.difficulty == DifficultyLevel.MEDIUM:
                self.blur_range = (0.5, 1.5)
                self.probability = 0.5
            else:  # HARD
                self.blur_range = (1.5, 3.0)
                self.probability = 0.7
            
            self.rng = random.Random(self.seed)
        
        def augment(self, image: Image.Image, **kwargs) -> Image.Image:
            """Apply blur augmentation."""
            if self.rng.random() < self.probability:
                radius = self.rng.uniform(*self.blur_range)
                return image.filter(ImageFilter.GaussianBlur(radius=radius))
            return image
        
        def get_augmentation_info(self) -> dict:
            """Return augmentation configuration info."""
            return {
                'type': 'CustomBlur',
                'difficulty': self.difficulty.value,
                'blur_range': self.blur_range,
                'probability': self.probability,
                'seed': self.seed
            }
    
    # Register custom augmentation
    AugmentationFactory.register_image_augmentation('custom_blur', CustomBlurAugmentation)
    
    # Use it
    custom_augmentor = AugmentationFactory.create_image_augmentation(
        augmentation_type='custom_blur',
        difficulty=DifficultyLevel.MEDIUM,
        seed=42
    )
    
    print(f"Custom augmentation registered and created!")
    print(f"Info: {custom_augmentor.get_augmentation_info()}")
    
    # Test it
    sample_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    augmented = custom_augmentor.augment(sample_image)
    print(f"Augmentation applied successfully!")
    print()


def example_5_dataset_with_augmentation():
    """Example 5: Using dataset with augmentation."""
    print("=" * 80)
    print("Example 5: Dataset with Augmentation")
    print("=" * 80)
    
    # This example shows how augmentation integrates with datasets
    print("Dataset augmentation integration:")
    print("  1. Create augmentor")
    print("  2. Create dataset")
    print("  3. Set augmentation: dataset.set_image_augmentation(augmentor.augment)")
    print("  4. During training, images are automatically augmented")
    print()
    
    # Example code (requires actual dataset):
    code_example = """
    # Create augmentor
    augmentor = AugmentationFactory.create_image_augmentation(
        augmentation_type='masked',
        difficulty=DifficultyLevel.EASY,
        patch_size=16
    )
    
    # Create dataset
    from dataset import ViVQADataset
    from transformers import AutoProcessor, AutoTokenizer
    
    vis_processor = AutoProcessor.from_pretrained('google/vit-base-patch16-224')
    text_processor = AutoTokenizer.from_pretrained('vinai/bartpho-syllable-base')
    
    dataset = ViVQADataset(
        ann_path='data/vivqa/train.csv',
        img_dir='data/vivqa/images',
        text_processor=text_processor,
        vis_processor=vis_processor,
        max_length=64
    )
    
    # Set augmentation
    dataset.set_image_augmentation(lambda img: augmentor.augment(img))
    
    # Now when you get items, images are automatically augmented
    sample = dataset[0]  # Image is augmented on-the-fly
    """
    
    print("Code example:")
    print(code_example)


def main():
    """Run all examples."""
    print("\n")
    print("*" * 80)
    print("VQA Training OOP Architecture Examples")
    print("*" * 80)
    print("\n")
    
    example_1_basic_augmentation()
    example_2_curriculum_learning()
    example_3_programmatic_training()
    example_4_custom_augmentation()
    example_5_dataset_with_augmentation()
    
    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
