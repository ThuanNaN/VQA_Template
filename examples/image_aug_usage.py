"""
Example usage of Image Augmentation with Curriculum Learning for VQA.

This script demonstrates how to use the MaskedImageAugmentation class
with curriculum learning to augment images for VQA training.
"""

import sys
sys.path.append('..')

from augmentation import (
    MaskedImageAugmentation,
    CurriculumLearningScheduler,
    DifficultyLevel
)
from PIL import Image
import os


def demonstrate_difficulty_levels():
    """Demonstrate augmentation at different difficulty levels."""
    print("="*60)
    print("Demonstrating Image Augmentation at Different Difficulty Levels")
    print("="*60)
    
    # Create a sample image (you can replace with actual image path)
    # For demonstration, we'll create a simple colored image
    from PIL import ImageDraw
    
    # Create a sample image with some patterns
    width, height = 224, 224
    sample_image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(sample_image)
    
    # Draw some patterns
    draw.rectangle([50, 50, 100, 100], fill='red')
    draw.rectangle([120, 50, 170, 100], fill='blue')
    draw.ellipse([50, 120, 100, 170], fill='green')
    draw.ellipse([120, 120, 170, 170], fill='yellow')
    
    print(f"\nOriginal image size: {sample_image.size}")
    
    # Test each difficulty level
    for difficulty in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]:
        print(f"\n{'-'*60}")
        print(f"Testing {difficulty.value.upper()} difficulty level")
        print(f"{'-'*60}")
        
        augmentor = MaskedImageAugmentation(difficulty=difficulty, seed=42)
        
        # Get augmentation info
        info = augmentor.get_augmentation_info()
        print(f"Configuration:")
        print(f"  - Mask ratio: {info['mask_ratio']:.2%}")
        print(f"  - Color jitter strength: {info['color_jitter_strength']}")
        print(f"  - Brightness range: {info['brightness_factor']}")
        print(f"  - Contrast range: {info['contrast_factor']}")
        print(f"  - Blur radius range: {info['blur_radius']}")
        print(f"  - Apply flip: {info['apply_flip']}")
        print(f"  - Crop scale range: {info['crop_scale']}")
        
        # Apply augmentation
        augmented = augmentor.augment(sample_image)
        print(f"  Augmented image size: {augmented.size}")


def demonstrate_curriculum_learning():
    """Demonstrate curriculum learning schedule."""
    print("\n" + "="*60)
    print("Demonstrating Curriculum Learning Schedule")
    print("="*60)
    
    total_epochs = 30
    scheduler = CurriculumLearningScheduler(total_epochs=total_epochs)
    
    schedule_info = scheduler.get_schedule_info()
    print(f"\nTraining Schedule for {schedule_info['total_epochs']} epochs:")
    print(f"  - Easy epochs: {schedule_info['easy_epochs']}")
    print(f"  - Medium epochs: {schedule_info['medium_epochs']}")
    print(f"  - Hard epochs: {schedule_info['hard_epochs']}")
    
    print("\nEpoch-by-epoch breakdown:")
    for schedule_line in schedule_info['schedule']:
        print(f"  {schedule_line}")
    
    # Show some example epochs
    print("\nExample difficulty levels for specific epochs:")
    for epoch in [0, 5, 9, 10, 15, 18, 19, 25, 29]:
        difficulty = scheduler.get_difficulty_for_epoch(epoch)
        print(f"  Epoch {epoch:2d}: {difficulty.value.upper()}")


def demonstrate_epoch_based_augmentation():
    """Demonstrate creating augmentors for different epochs."""
    print("\n" + "="*60)
    print("Demonstrating Epoch-based Augmentation")
    print("="*60)
    
    # Create curriculum schedule
    total_epochs = 30
    scheduler = CurriculumLearningScheduler(total_epochs=total_epochs)
    
    # Create a sample image
    width, height = 224, 224
    sample_image = Image.new('RGB', (width, height), color='lightblue')
    
    print("\nCreating augmentors for different epochs:")
    for epoch in [0, 9, 18, 29]:  # Sample epochs from each difficulty level
        difficulty = scheduler.get_difficulty_for_epoch(epoch)
        augmentor = MaskedImageAugmentation(difficulty=difficulty, seed=42)
        
        print(f"\n  Epoch {epoch} ({difficulty.value.upper()}):")
        info = augmentor.get_augmentation_info()
        print(f"    Mask ratio: {info['mask_ratio']:.2%}")
        
        # Apply augmentation
        augmented = augmentor.augment(sample_image)
        print(f"    Augmentation applied successfully")


def demonstrate_selective_augmentations():
    """Demonstrate applying selective augmentation techniques."""
    print("\n" + "="*60)
    print("Demonstrating Selective Augmentations")
    print("="*60)
    
    # Create sample image
    width, height = 224, 224
    sample_image = Image.new('RGB', (width, height), color='coral')
    
    augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.MEDIUM, seed=42)
    
    print("\nApplying different combinations of augmentations:")
    
    # Only masking
    print("\n  1. Only masking (no other augmentations):")
    masked = augmentor.augment(
        sample_image,
        apply_masking=True,
        apply_color_jitter=False,
        apply_blur=False,
        apply_brightness=False,
        apply_contrast=False
    )
    print("     Applied successfully")
    
    # Only color transformations
    print("\n  2. Only color transformations (no masking):")
    color_aug = augmentor.augment(
        sample_image,
        apply_masking=False,
        apply_color_jitter=True,
        apply_brightness=True,
        apply_contrast=True,
        apply_blur=False
    )
    print("     Applied successfully")
    
    # Only blur
    print("\n  3. Only blur:")
    blurred = augmentor.augment(
        sample_image,
        apply_masking=False,
        apply_color_jitter=False,
        apply_blur=True,
        apply_brightness=False,
        apply_contrast=False
    )
    print("     Applied successfully")
    
    # Full augmentation
    print("\n  4. Full augmentation (all techniques):")
    full_aug = augmentor.augment(sample_image)
    print("     Applied successfully")


def demonstrate_custom_schedule():
    """Demonstrate creating a custom curriculum schedule."""
    print("\n" + "="*60)
    print("Demonstrating Custom Curriculum Schedule")
    print("="*60)
    
    # Custom schedule: longer easy period, shorter hard period
    total_epochs = 50
    scheduler = CurriculumLearningScheduler(
        total_epochs=total_epochs,
        easy_epochs=20,    # 40% of training
        medium_epochs=20,  # 40% of training
        hard_epochs=10     # 20% of training
    )
    
    schedule_info = scheduler.get_schedule_info()
    print(f"\nCustom Schedule for {schedule_info['total_epochs']} epochs:")
    print(f"  - Easy epochs: {schedule_info['easy_epochs']} (40%)")
    print(f"  - Medium epochs: {schedule_info['medium_epochs']} (40%)")
    print(f"  - Hard epochs: {schedule_info['hard_epochs']} (20%)")
    
    print("\nSchedule breakdown:")
    for schedule_line in schedule_info['schedule']:
        print(f"  {schedule_line}")


def demonstrate_with_real_image():
    """Demonstrate augmentation with real images from ViVQA dataset."""
    print("\n" + "="*60)
    print("Demonstrating with Real Images from ViVQA Dataset")
    print("="*60)
    
    # Path to ViVQA images directory
    vivqa_images_dir = "../data/vivqa/images"
    
    if not os.path.exists(vivqa_images_dir):
        print("\nViVQA images directory not found.")
        print("Skipping real image demonstration.")
        return
    
    # Get first 3 images from the directory
    image_files = sorted([f for f in os.listdir(vivqa_images_dir) if f.endswith('.jpg')])[:3]
    
    if not image_files:
        print("\nNo images found in ViVQA directory.")
        print("Skipping real image demonstration.")
        return
    
    # Create output directory for augmented images
    output_dir = "../runs/image_aug_demo"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Process each sample image
    for img_file in image_files:
        image_path = os.path.join(vivqa_images_dir, img_file)
        print(f"\n{'-'*60}")
        print(f"Processing: {img_file}")
        print(f"{'-'*60}")
        
        try:
            # Load the image
            image = Image.open(image_path).convert('RGB')
            print(f"Original image size: {image.size}")
            
            # Save original image
            img_name = os.path.splitext(img_file)[0]
            original_output = os.path.join(output_dir, f"{img_name}_original.jpg")
            image.save(original_output)
            print(f"Saved original: {original_output}")
            
            # Apply augmentation at different difficulty levels
            for difficulty in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]:
                print(f"\n  Applying {difficulty.value.upper()} augmentation:")
                
                augmentor = MaskedImageAugmentation(difficulty=difficulty, seed=42)
                
                # Get augmentation info
                info = augmentor.get_augmentation_info()
                print(f"    - Mask ratio: {info['mask_ratio']:.2%}")
                print(f"    - Color jitter: {info['color_jitter_strength']}")
                
                # Apply augmentation
                augmented = augmentor.augment(image)
                
                # Save augmented image
                output_path = os.path.join(output_dir, f"{img_name}_{difficulty.value}.jpg")
                augmented.save(output_path)
                print(f"    - Saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"All augmented images saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("IMAGE AUGMENTATION WITH CURRICULUM LEARNING")
    print("Masked Autoencoder-Inspired Augmentation for VQA")
    print("="*60)
    
    # Run all demonstrations
    demonstrate_difficulty_levels()
    demonstrate_curriculum_learning()
    demonstrate_epoch_based_augmentation()
    demonstrate_selective_augmentations()
    demonstrate_custom_schedule()
    demonstrate_with_real_image()
    
    print("\n" + "="*60)
    print("All demonstrations completed successfully!")
    print("="*60)
    print("\nKey Takeaways:")
    print("  1. Use DifficultyLevel (EASY, MEDIUM, HARD) to control augmentation intensity")
    print("  2. CurriculumLearningScheduler manages progression during training")
    print("  3. MaskedImageAugmentation provides flexible, configurable augmentations")
    print("  4. Selective augmentations can be applied as needed")
    print("  5. Integration with training loop is straightforward")
    print("\nNext Steps:")
    print("  - Integrate with your VQA training pipeline")
    print("  - Experiment with different curriculum schedules")
    print("  - Tune augmentation parameters for your dataset")
    print("  - Monitor model performance across difficulty levels")
    print()


if __name__ == "__main__":
    main()
