"""
Integration example: Using Image Augmentation with VQA Dataset

This script demonstrates how to integrate the MaskedImageAugmentation
with the existing VQA dataset pipeline and training workflow.
"""

import sys
sys.path.append('..')

from augmentation import (
    MaskedImageAugmentation,
    CurriculumLearningScheduler,
    DifficultyLevel,
    create_augmentor_for_epoch
)
from PIL import Image
import torch
from torch.utils.data import Dataset


class AugmentedVQADataset(Dataset):
    """
    Example VQA Dataset with Image Augmentation support.
    
    This class demonstrates how to integrate MaskedImageAugmentation
    into a VQA dataset for training with curriculum learning.
    
    Args:
        base_dataset: Original VQA dataset
        difficulty: Difficulty level for augmentation (EASY, MEDIUM, HARD)
        patch_size: Size of patches for masking
        apply_augmentation: Whether to apply augmentation (set False for validation)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        base_dataset,
        difficulty: DifficultyLevel = DifficultyLevel.EASY,
        patch_size: int = 16,
        apply_augmentation: bool = True,
        seed: int = None
    ):
        self.base_dataset = base_dataset
        self.apply_augmentation = apply_augmentation
        
        if apply_augmentation:
            self.augmentor = MaskedImageAugmentation(
                difficulty=difficulty,
                patch_size=patch_size,
                seed=seed
            )
        else:
            self.augmentor = None
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get item from base dataset
        item = self.base_dataset[idx]
        
        # If augmentation is enabled, apply it to the image
        if self.apply_augmentation and self.augmentor is not None:
            # Assume item has an 'image' key with PIL Image
            # This would need to be adapted based on actual dataset structure
            if 'image' in item and isinstance(item['image'], Image.Image):
                item['image'] = self.augmentor.augment(item['image'])
        
        return item
    
    def update_difficulty(self, difficulty: DifficultyLevel):
        """Update the augmentation difficulty level."""
        if self.augmentor is not None:
            self.augmentor = MaskedImageAugmentation(
                difficulty=difficulty,
                patch_size=self.augmentor.patch_size,
                seed=self.augmentor.seed
            )


class CurriculumVQATrainer:
    """
    Example trainer with Curriculum Learning support.
    
    Demonstrates how to progressively increase augmentation difficulty
    during training using the CurriculumLearningScheduler.
    """
    
    def __init__(
        self,
        train_dataset,
        val_dataset,
        total_epochs: int = 30,
        easy_epochs: int = None,
        medium_epochs: int = None,
        hard_epochs: int = None
    ):
        self.base_train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Create curriculum scheduler
        self.scheduler = CurriculumLearningScheduler(
            total_epochs=total_epochs,
            easy_epochs=easy_epochs,
            medium_epochs=medium_epochs,
            hard_epochs=hard_epochs
        )
        
        # Initialize with EASY difficulty
        self.train_dataset = AugmentedVQADataset(
            base_dataset=train_dataset,
            difficulty=DifficultyLevel.EASY,
            apply_augmentation=True
        )
        
        print("Curriculum Learning Schedule:")
        schedule_info = self.scheduler.get_schedule_info()
        for line in schedule_info['schedule']:
            print(f"  {line}")
    
    def train_epoch(self, epoch: int):
        """
        Train for one epoch with appropriate difficulty level.
        
        Args:
            epoch: Current epoch number (0-indexed)
        """
        # Get difficulty for current epoch
        difficulty = self.scheduler.get_difficulty_for_epoch(epoch)
        
        # Update dataset augmentation difficulty
        self.train_dataset.update_difficulty(difficulty)
        
        print(f"\nEpoch {epoch}: Training with {difficulty.value.upper()} augmentation")
        
        # Your training loop would go here
        # For demonstration, we just show the configuration
        aug_info = self.train_dataset.augmentor.get_augmentation_info()
        print(f"  Mask ratio: {aug_info['mask_ratio']:.2%}")
        print(f"  Color jitter: {aug_info['color_jitter_strength']}")
        print(f"  Brightness range: {aug_info['brightness_factor']}")
        
        # Pseudo training loop
        # for batch in train_loader:
        #     # Forward pass
        #     # Backward pass
        #     # Update weights
        #     pass
        
        return {"train_loss": 0.5}  # Placeholder
    
    def validate(self, epoch: int):
        """Run validation (no augmentation)."""
        print(f"  Validating...")
        
        # Validation uses original images without augmentation
        # for batch in val_loader:
        #     # Forward pass
        #     # Calculate metrics
        #     pass
        
        return {"val_loss": 0.4, "val_acc": 0.75}  # Placeholder
    
    def train(self):
        """Full training loop with curriculum learning."""
        total_epochs = self.scheduler.total_epochs
        
        for epoch in range(total_epochs):
            # Train with appropriate difficulty
            train_metrics = self.train_epoch(epoch)
            
            # Validate (no augmentation)
            val_metrics = self.validate(epoch)
            
            print(f"  Results: train_loss={train_metrics['train_loss']:.4f}, "
                  f"val_loss={val_metrics['val_loss']:.4f}, "
                  f"val_acc={val_metrics['val_acc']:.4f}")


def demonstrate_integration():
    """Demonstrate integration with VQA dataset."""
    print("="*70)
    print("INTEGRATION EXAMPLE: Image Augmentation with VQA Dataset")
    print("="*70)
    
    # Create a mock dataset for demonstration
    class MockVQADataset:
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Return a mock item
            return {
                'image': Image.new('RGB', (224, 224), color='blue'),
                'question': f"Sample question {idx}",
                'answer': f"Answer {idx % 10}"
            }
    
    # Create datasets
    train_dataset = MockVQADataset(size=1000)
    val_dataset = MockVQADataset(size=200)
    
    print("\n1. Creating Curriculum Learning Trainer")
    print("-" * 70)
    
    trainer = CurriculumVQATrainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        total_epochs=12,  # Shorter for demo
        easy_epochs=4,
        medium_epochs=4,
        hard_epochs=4
    )
    
    print("\n2. Simulating Training with Curriculum Learning")
    print("-" * 70)
    
    # Simulate training for a few epochs
    for epoch in [0, 4, 8, 11]:
        trainer.train_epoch(epoch)


def demonstrate_dataloader_integration():
    """Demonstrate integration with PyTorch DataLoader."""
    print("\n" + "="*70)
    print("DATALOADER INTEGRATION EXAMPLE")
    print("="*70)
    
    from torch.utils.data import DataLoader
    
    # Create a simple dataset
    class SimpleVQADataset:
        def __init__(self, size=50):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Convert image to tensor for DataLoader compatibility
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            
            image = Image.new('RGB', (224, 224), color='green')
            return {
                'image': to_tensor(image),  # Convert to tensor
                'question_ids': torch.randint(0, 1000, (20,)),
                'label': torch.tensor(idx % 10)
            }
    
    base_dataset = SimpleVQADataset(size=50)
    
    # Create augmented dataset for each difficulty level
    print("\nCreating DataLoaders for different difficulty levels:")
    print("  (Note: In real use, convert PIL images to tensors in __getitem__)")
    
    for difficulty in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]:
        print(f"\n  {difficulty.value.upper()} Difficulty:")
        
        aug_dataset = AugmentedVQADataset(
            base_dataset=base_dataset,
            difficulty=difficulty,
            apply_augmentation=True,
            seed=42
        )
        
        dataloader = DataLoader(
            aug_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0
        )
        
        # Get a batch
        batch = next(iter(dataloader))
        print(f"    Batch size: {len(batch['label'])}")
        print(f"    Image shape: {batch['image'].shape}")
        print(f"    Question IDs shape: {batch['question_ids'].shape}")
        print(f"    Labels shape: {batch['label'].shape}")


def demonstrate_epoch_progression():
    """Demonstrate how augmentation changes across epochs."""
    print("\n" + "="*70)
    print("EPOCH PROGRESSION DEMONSTRATION")
    print("="*70)
    
    # Create scheduler
    scheduler = CurriculumLearningScheduler(total_epochs=20)
    
    print("\nAugmentation parameters across epochs:")
    print(f"{'Epoch':>6} | {'Difficulty':>10} | {'Mask Ratio':>12} | {'Color Jitter':>14}")
    print("-" * 60)
    
    for epoch in range(0, 20, 2):  # Every other epoch
        difficulty = scheduler.get_difficulty_for_epoch(epoch)
        augmentor = create_augmentor_for_epoch(epoch, scheduler)
        info = augmentor.get_augmentation_info()
        
        print(f"{epoch:6d} | {difficulty.value:>10} | {info['mask_ratio']:>11.1%} | "
              f"{info['color_jitter_strength']:>14.2f}")


def demonstrate_selective_augmentation():
    """Demonstrate using selective augmentation during training."""
    print("\n" + "="*70)
    print("SELECTIVE AUGMENTATION DEMONSTRATION")
    print("="*70)
    
    # Sometimes you might want to apply only specific augmentations
    print("\nApplying different augmentation strategies:")
    
    test_image = Image.new('RGB', (224, 224), color='red')
    augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.MEDIUM)
    
    strategies = [
        ("Masking only", {
            'apply_masking': True,
            'apply_color_jitter': False,
            'apply_blur': False,
            'apply_brightness': False,
            'apply_contrast': False
        }),
        ("Color augmentation only", {
            'apply_masking': False,
            'apply_color_jitter': True,
            'apply_brightness': True,
            'apply_contrast': True,
            'apply_blur': False
        }),
        ("Full augmentation", {
            'apply_masking': True,
            'apply_color_jitter': True,
            'apply_blur': True,
            'apply_brightness': True,
            'apply_contrast': True
        })
    ]
    
    for strategy_name, kwargs in strategies:
        augmented = augmentor.augment(test_image, **kwargs)
        print(f"  ✓ {strategy_name}: Applied successfully")


def main():
    """Run all integration demonstrations."""
    print("\n" + "="*70)
    print("VQA IMAGE AUGMENTATION - INTEGRATION EXAMPLES")
    print("="*70)
    
    demonstrate_integration()
    demonstrate_dataloader_integration()
    demonstrate_epoch_progression()
    demonstrate_selective_augmentation()
    
    print("\n" + "="*70)
    print("INTEGRATION EXAMPLES COMPLETED")
    print("="*70)
    
    print("\nKey Integration Points:")
    print("  1. Wrap your dataset with AugmentedVQADataset")
    print("  2. Use CurriculumLearningScheduler to manage difficulty progression")
    print("  3. Update augmentation difficulty at the start of each epoch")
    print("  4. Keep validation without augmentation for fair evaluation")
    print("  5. Use selective augmentations for specific training strategies")
    
    print("\nNext Steps for Real Integration:")
    print("  - Modify __getitem__ to work with your actual dataset structure")
    print("  - Integrate with your training loop")
    print("  - Experiment with different curriculum schedules")
    print("  - Monitor how augmentation difficulty affects learning")
    print("  - Log augmentation parameters with your experiment tracker (wandb, etc.)")
    print()


if __name__ == "__main__":
    main()
