# MAE-Inspired Patch Masking Augmentation

Comprehensive documentation for the Masked Autoencoder (MAE) inspired image augmentation system with Curriculum Learning support.

## 📋 Table of Contents

- [Overview](#overview)
- [Inspiration & Background](#inspiration--background)
- [Architecture](#architecture)
- [Curriculum Learning Strategy](#curriculum-learning-strategy)
- [Augmentation Techniques](#augmentation-techniques)
- [Usage Examples](#usage-examples)
- [Configuration Reference](#configuration-reference)
- [Best Practices](#best-practices)
- [Performance Considerations](#performance-considerations)

## Overview

The **MaskedImageAugmentation** system implements a sophisticated image augmentation framework inspired by [Masked Autoencoders Are Scalable Vision Learners (CVPR 2022)](https://arxiv.org/abs/2111.06377). It combines patch-based masking with curriculum learning to progressively train VQA models from easy to hard samples.

### Key Features

✅ **MAE-Inspired Masking** - Random patch masking following the MAE approach
✅ **Curriculum Learning** - Progressive difficulty (EASY → MEDIUM → HARD)
✅ **Multiple Augmentation Types** - Masking, color jitter, blur, brightness, contrast
✅ **Configurable Parameters** - Flexible control over all augmentation aspects
✅ **Reproducible** - Seed-based random generation
✅ **Epoch-Aware** - Automatic difficulty adjustment based on training progress

## Inspiration & Background

### Masked Autoencoders (MAE)

The MAE paper introduced a powerful self-supervised learning approach where:

1. **Random patches are masked** (typically 75% of the image)
2. **Encoder processes visible patches** only
3. **Decoder reconstructs masked patches** from encoded representation

This approach teaches the model to:

- Understand spatial relationships
- Learn robust visual representations
- Handle incomplete information (crucial for VQA)

### Adaptation for VQA

Our implementation adapts MAE's masking strategy for supervised VQA training:

- **Variable masking ratios** based on training difficulty
- **Combined with other augmentations** for richer diversity
- **Progressive difficulty** to prevent overwhelming the model
- **Task-specific tuning** optimized for Vietnamese VQA datasets

## Architecture

### Class Hierarchy

```plaintext
BaseImageAugmentation (abstract)
    ↓
MaskedImageAugmentation
    ↓ uses
CurriculumLearningScheduler
```

### Core Components

```python
MaskedImageAugmentation
├── Patch Masking (MAE-inspired)
│   ├── Random patch selection
│   ├── Configurable mask ratio
│   └── Gray color (128) masking
│
├── Color Transformations
│   ├── Color jitter
│   ├── Brightness adjustment
│   └── Contrast adjustment
│
├── Spatial Transformations
│   ├── Gaussian blur
│   ├── Random crop
│   └── Horizontal flip
│
└── Curriculum Learning
    ├── Difficulty-based parameters
    ├── Epoch-aware scheduling
    └── Progressive augmentation
```

## Curriculum Learning Strategy

### Three-Stage Progressive Training

The system implements a three-stage curriculum that gradually increases augmentation difficulty:

#### 🟢 Stage 1: EASY (33% of training)

**Goal**: Build foundational understanding with minimal distortion

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Mask Ratio | 15% | Light masking to maintain most visual information |
| Color Jitter | 0.1 | Minimal color variation |
| Brightness | (0.9, 1.1) | Slight brightness changes |
| Contrast | (0.9, 1.1) | Slight contrast changes |
| Blur Radius | (0.1, 0.3) | Minimal blurring |
| Flip | ❌ | No flipping to preserve orientation |
| Crop Scale | (0.95, 1.0) | Minimal cropping |

**Example**: If total_epochs=30, EASY spans epochs 0-9

#### 🟡 Stage 2: MEDIUM (33% of training)

**Goal**: Introduce moderate challenges to improve robustness

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Mask Ratio | 50% | Half of patches masked (challenging but learnable) |
| Color Jitter | 0.3 | Moderate color variation |
| Brightness | (0.7, 1.3) | Noticeable brightness changes |
| Contrast | (0.7, 1.3) | Noticeable contrast changes |
| Blur Radius | (0.5, 1.5) | Moderate blurring |
| Flip | ✅ | Random horizontal flips |
| Crop Scale | (0.8, 1.0) | Moderate cropping |

**Example**: If total_epochs=30, MEDIUM spans epochs 10-19

#### 🔴 Stage 3: HARD (34% of training)

**Goal**: Maximize generalization with aggressive augmentation

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Mask Ratio | 75% | Heavy masking (MAE paper standard) |
| Color Jitter | 0.5 | Strong color variation |
| Brightness | (0.5, 1.5) | Extreme brightness changes |
| Contrast | (0.5, 1.5) | Extreme contrast changes |
| Blur Radius | (1.0, 3.0) | Strong blurring |
| Flip | ✅ | Random horizontal flips |
| Crop Scale | (0.7, 1.0) | Aggressive cropping |

**Example**: If total_epochs=30, HARD spans epochs 20-29

### Visual Progression

```plaintext
EASY                  MEDIUM                HARD
────────────────────────────────────────────────────
█████████████████  ██████░░░░░░░░░░  ███░░░░░░░░░░░
█████████████████  ██████░░░░░░░░░░  ███░░░░░░░░░░░
█████████████████  ██████░░░░░░░░░░  ███░░░░░░░░░░░
                                      
15% masked           50% masked         75% masked
Minimal changes      Moderate changes   Aggressive changes
```

## Augmentation Techniques

### 1. Random Patch Masking (MAE-Inspired)

**Core Technique** - The primary augmentation method

```python
def _random_masking(self, image: Image.Image) -> Image.Image:
    """
    Apply random masking inspired by Masked Autoencoders.
    
    Process:
    1. Divide image into patches (default: 16x16)
    2. Randomly select patches to mask
    3. Fill masked patches with gray (128, 128, 128)
    """
```

**How it works**:

1. **Patchification**: Divide image into non-overlapping patches
   - Default patch size: 16×16 pixels
   - For 224×224 image: 14×14 = 196 patches

2. **Random Selection**: Choose patches to mask
   - EASY: 15% of patches (≈30 patches)
   - MEDIUM: 50% of patches (≈98 patches)
   - HARD: 75% of patches (≈147 patches)

3. **Masking**: Fill selected patches with gray color (128, 128, 128)

**Why gray (128)?**

- Neutral value in RGB space
- Doesn't bias model toward any color
- Clearly distinguishes masked from visible regions

**Visual Example**:

```plaintext
Original Image (224×224)        Masked Image (50% mask ratio)
┌─────────────────────┐         ┌─────────────────────┐
│ [Person] [Objects]  │         │ ░░░░░░░ [Objects]  │
│                     │   →     │                     │
│ [Background]        │         │ ░░░░░░░ ░░░░░░░   │
└─────────────────────┘         └─────────────────────┘
                                ░ = Masked patches (gray)
```

### 2. Color Jittering

**Purpose**: Improve color invariance

```python
def _color_jitter(self, image: Image.Image) -> Image.Image:
    """Apply random hue/saturation shifts."""
```

- **Method**: Random hue shift using PIL's Color enhancer
- **Strength**: Varies by difficulty (0.1 → 0.3 → 0.5)
- **Effect**: Model learns to focus on shapes, not just colors

### 3. Brightness Adjustment

**Purpose**: Handle varying lighting conditions

```python
def _adjust_brightness(self, image: Image.Image) -> Image.Image:
    """Randomly adjust image brightness."""
```

- **EASY**: ±10% variation
- **MEDIUM**: ±30% variation
- **HARD**: ±50% variation

### 4. Contrast Adjustment

**Purpose**: Improve robustness to contrast changes

```python
def _adjust_contrast(self, image: Image.Image) -> Image.Image:
    """Randomly adjust image contrast."""
```

- **EASY**: ±10% variation
- **MEDIUM**: ±30% variation
- **HARD**: ±50% variation

### 5. Gaussian Blur

**Purpose**: Simulate out-of-focus or low-quality images

```python
def _gaussian_blur(self, image: Image.Image) -> Image.Image:
    """Apply random Gaussian blur."""
```

- **EASY**: Radius 0.1-0.3 (minimal blur)
- **MEDIUM**: Radius 0.5-1.5 (moderate blur)
- **HARD**: Radius 1.0-3.0 (strong blur)

### 6. Random Cropping

**Purpose**: Learn to handle partial objects/scenes

```python
def _random_crop(self, image: Image.Image) -> Image.Image:
    """Random crop and resize back to original size."""
```

- **EASY**: 95-100% of image retained
- **MEDIUM**: 80-100% of image retained
- **HARD**: 70-100% of image retained

### 7. Horizontal Flipping

**Purpose**: Data augmentation for left-right symmetry

- **EASY**: Disabled (preserve orientation)
- **MEDIUM**: 50% probability
- **HARD**: 50% probability

## Usage Examples

### Basic Usage

```python
from augmentation import MaskedImageAugmentation, DifficultyLevel
from PIL import Image

# Load image
image = Image.open("path/to/image.jpg")

# Create augmentor with MEDIUM difficulty
augmentor = MaskedImageAugmentation(
    difficulty=DifficultyLevel.MEDIUM,
    patch_size=16,
    seed=42  # For reproducibility
)

# Apply augmentation
augmented_image = augmentor.augment(image)

# Get configuration info
config = augmentor.get_augmentation_info()
print(f"Mask ratio: {config['mask_ratio']:.2%}")
```

### Curriculum Learning Integration

```python
from augmentation import (
    MaskedImageAugmentation,
    CurriculumLearningScheduler,
    create_augmentor_for_epoch
)

# Setup curriculum schedule
total_epochs = 30
scheduler = CurriculumLearningScheduler(total_epochs=total_epochs)

# Training loop
for epoch in range(total_epochs):
    # Create augmentor for current epoch
    augmentor = create_augmentor_for_epoch(
        epoch=epoch,
        scheduler=scheduler,
        patch_size=16,
        seed=42
    )
    
    # Get current difficulty
    difficulty = scheduler.get_difficulty_for_epoch(epoch)
    print(f"Epoch {epoch}: {difficulty.value} difficulty")
    
    # Your training code here
    for batch in dataloader:
        images = batch['images']
        # Augment images
        augmented = [augmentor.augment(img) for img in images]
        # ... training logic
```

### Custom Curriculum Schedule

```python
from augmentation import CurriculumLearningScheduler

# Create custom schedule (60 total epochs)
scheduler = CurriculumLearningScheduler(
    total_epochs=60,
    easy_epochs=20,      # First 20 epochs: EASY
    medium_epochs=20,    # Next 20 epochs: MEDIUM
    hard_epochs=20       # Last 20 epochs: HARD
)

# View schedule
info = scheduler.get_schedule_info()
print(f"Easy: epochs {info['easy_range']}")
print(f"Medium: epochs {info['medium_range']}")
print(f"Hard: epochs {info['hard_range']}")
```

### Selective Augmentation

```python
# Create augmentor
augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.HARD)

# Apply only masking and blur (disable other augmentations)
augmented = augmentor.augment(
    image,
    apply_masking=True,
    apply_blur=True,
    apply_color_jitter=False,
    apply_brightness=False,
    apply_contrast=False,
    apply_crop=False,
    apply_flip=False
)
```

### Custom Mask Ratio

```python
# Override default mask ratio
augmentor = MaskedImageAugmentation(
    difficulty=DifficultyLevel.MEDIUM,
    mask_ratio=0.35,  # Custom 35% masking
    patch_size=16
)

# Check configuration
info = augmentor.get_augmentation_info()
print(f"Using {info['mask_ratio']:.1%} mask ratio")
```

### Integration with Dataset

```python
from dataset import BaseVQADataset
from augmentation import MaskedImageAugmentation

class MyVQADataset(BaseVQADataset):
    def __init__(self, *args, augmentor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentor = augmentor
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        # Apply augmentation if available
        if self.augmentor is not None:
            item['image'] = self.augmentor.augment(item['image'])
        
        return item

# Usage
augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.MEDIUM)
dataset = MyVQADataset(
    dataset_name='vivqa',
    split='train',
    augmentor=augmentor
)
```

## Configuration Reference

### Constructor Parameters

```python
MaskedImageAugmentation(
    difficulty: Union[DifficultyLevel, str] = DifficultyLevel.EASY,
    patch_size: int = 16,
    mask_ratio: Optional[float] = None,
    seed: Optional[int] = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `difficulty` | DifficultyLevel or str | EASY | Augmentation difficulty level |
| `patch_size` | int | 16 | Size of patches for masking |
| `mask_ratio` | float or None | None | Custom mask ratio (overrides difficulty default) |
| `seed` | int or None | None | Random seed for reproducibility |

### Augment Method Parameters

```python
augmentor.augment(
    image: Image.Image,
    apply_masking: bool = True,
    apply_color_jitter: bool = True,
    apply_blur: bool = True,
    apply_brightness: bool = True,
    apply_contrast: bool = True,
    apply_crop: bool = False,
    apply_flip: bool = None
) -> Image.Image
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | PIL.Image | Required | Input image to augment |
| `apply_masking` | bool | True | Apply random patch masking |
| `apply_color_jitter` | bool | True | Apply color jittering |
| `apply_blur` | bool | True | Apply Gaussian blur |
| `apply_brightness` | bool | True | Adjust brightness |
| `apply_contrast` | bool | True | Adjust contrast |
| `apply_crop` | bool | False | Apply random cropping |
| `apply_flip` | bool | None | Apply horizontal flip (uses difficulty default if None) |

### Difficulty Level Presets

| Difficulty | Mask Ratio | Color Jitter | Brightness | Contrast | Blur Radius | Flip | Crop Scale |
|------------|-----------|-------------|-----------|---------|------------|------|-----------|
| EASY | 15% | 0.1 | (0.9, 1.1) | (0.9, 1.1) | (0.1, 0.3) | ❌ | (0.95, 1.0) |
| MEDIUM | 50% | 0.3 | (0.7, 1.3) | (0.7, 1.3) | (0.5, 1.5) | ✅ | (0.8, 1.0) |
| HARD | 75% | 0.5 | (0.5, 1.5) | (0.5, 1.5) | (1.0, 3.0) | ✅ | (0.7, 1.0) |

## Best Practices

### 1. Start with Curriculum Learning

✅ **DO**: Use curriculum learning for new models
```python
scheduler = CurriculumLearningScheduler(total_epochs=30)
for epoch in range(30):
    augmentor = create_augmentor_for_epoch(epoch, scheduler)
```

❌ **DON'T**: Jump directly to HARD difficulty
```python
# This might overwhelm the model in early training
augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.HARD)
```

### 2. Set Random Seeds for Reproducibility

✅ **DO**: Use seeds for reproducible experiments
```python
augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.MEDIUM, seed=42)
```

### 3. Adjust Patch Size Based on Image Resolution

```python
# For 224×224 images
augmentor = MaskedImageAugmentation(patch_size=16)  # 14×14 patches

# For 384×384 images
augmentor = MaskedImageAugmentation(patch_size=24)  # 16×16 patches

# For 512×512 images
augmentor = MaskedImageAugmentation(patch_size=32)  # 16×16 patches
```

### 4. Balance Augmentation Strength

✅ **DO**: Start conservative, increase gradually
```python
# Good progression over 30 epochs
scheduler = CurriculumLearningScheduler(
    total_epochs=30,
    easy_epochs=10,
    medium_epochs=10,
    hard_epochs=10
)
```

❌ **DON'T**: Rush to high difficulty
```python
# Bad: Mostly hard augmentation
scheduler = CurriculumLearningScheduler(
    total_epochs=30,
    easy_epochs=3,
    medium_epochs=5,
    hard_epochs=22
)
```

### 5. Monitor Training Metrics

Track how augmentation affects learning:

```python
import wandb

for epoch in range(total_epochs):
    augmentor = create_augmentor_for_epoch(epoch, scheduler)
    
    # Log augmentation config
    config = augmentor.get_augmentation_info()
    wandb.log({
        'epoch': epoch,
        'difficulty': config['difficulty'],
        'mask_ratio': config['mask_ratio'],
        'train_loss': train_loss,
        'val_accuracy': val_accuracy
    })
```

### 6. Use Selective Augmentation for Debugging

```python
# Debug by testing one augmentation at a time
augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.MEDIUM)

# Test only masking
masked_only = augmentor.augment(
    image,
    apply_masking=True,
    apply_color_jitter=False,
    apply_blur=False,
    apply_brightness=False,
    apply_contrast=False
)
```

### 7. Validate Augmentation Quality

```python
from PIL import Image
import matplotlib.pyplot as plt

def visualize_augmentation(original_image, augmentor, num_samples=5):
    """Visualize augmentation effects."""
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(15, 3))
    
    # Show original
    axes[0].imshow(original_image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show augmented versions
    for i in range(num_samples):
        augmented = augmentor.augment(original_image)
        axes[i + 1].imshow(augmented)
        axes[i + 1].set_title(f'Aug {i+1}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Test augmentation
augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.MEDIUM, seed=None)
visualize_augmentation(image, augmentor, num_samples=5)
```

## Performance Considerations

### Computational Cost

| Operation | Relative Cost | Notes |
|-----------|--------------|-------|
| Random Masking | Low | Numpy array operations, very fast |
| Color Jitter | Medium | PIL enhancement operations |
| Brightness/Contrast | Medium | PIL enhancement operations |
| Gaussian Blur | High | Convolution operation, slower |
| Random Crop | Low | Fast cropping and resizing |
| Flip | Very Low | Simple transpose operation |

### Optimization Tips

#### 1. Batch Augmentation

```python
# Efficient: Reuse augmentor for batch
augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.MEDIUM)
augmented_batch = [augmentor.augment(img) for img in images]
```

#### 2. Disable Expensive Operations for Large Datasets

```python
# For very large datasets, consider disabling blur
augmented = augmentor.augment(image, apply_blur=False)
```

#### 3. Use DataLoader Workers

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Parallel augmentation
    pin_memory=True
)
```

#### 4. Cache Augmented Images (if memory allows)

```python
class CachedAugmentedDataset(BaseVQADataset):
    def __init__(self, *args, augmentor=None, cache_size=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentor = augmentor
        self.cache = {}
        self.cache_size = cache_size
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        item = super().__getitem__(idx)
        if self.augmentor:
            item['image'] = self.augmentor.augment(item['image'])
        
        if len(self.cache) < self.cache_size:
            self.cache[idx] = item
        
        return item
```

### Memory Usage

Typical memory footprint per augmentation:

- **Original Image** (224×224×3): ~150 KB
- **Augmented Image** (224×224×3): ~150 KB
- **Temporary Arrays**: ~50 KB
- **Total per image**: ~350 KB

For a batch of 32 images: ~11 MB

## Advanced Topics

### Custom Difficulty Levels

```python
from augmentation import BaseImageAugmentation, DifficultyLevel

class CustomAugmentation(BaseImageAugmentation):
    def _configure_parameters(self):
        if self.difficulty == DifficultyLevel.EASY:
            self.mask_ratio = 0.10  # Even lighter masking
            # ... other parameters
        elif self.difficulty == DifficultyLevel.MEDIUM:
            self.mask_ratio = 0.40  # Custom medium
            # ... other parameters
        else:  # HARD
            self.mask_ratio = 0.80  # Even harder
            # ... other parameters
```

### Block Masking Strategy

For future enhancement, consider block masking:

```python
# Potential enhancement: Mask contiguous blocks instead of random patches
def _block_masking(self, image, block_size=4):
    """Mask contiguous blocks of patches."""
    # Implementation would group adjacent patches
    # More challenging than random masking
    pass
```

### Adaptive Masking

Adjust masking based on model performance:

```python
class AdaptiveAugmentor:
    def __init__(self, initial_difficulty=DifficultyLevel.EASY):
        self.augmentor = MaskedImageAugmentation(difficulty=initial_difficulty)
    
    def update_difficulty(self, val_accuracy):
        """Increase difficulty if model is performing well."""
        if val_accuracy > 0.8 and self.augmentor.difficulty == DifficultyLevel.EASY:
            self.augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.MEDIUM)
        elif val_accuracy > 0.9 and self.augmentor.difficulty == DifficultyLevel.MEDIUM:
            self.augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.HARD)
```

## Troubleshooting

### Issue: Model Not Learning

**Symptoms**: Loss not decreasing, accuracy stays low

**Solutions**:

1. Start with EASY difficulty
2. Reduce mask ratio: `mask_ratio=0.1`
3. Disable blur initially: `apply_blur=False`
4. Check if augmentation is too aggressive

### Issue: Overfitting

**Symptoms**: Training accuracy high, validation accuracy low

**Solutions**:

1. Increase mask ratio
2. Use HARD difficulty earlier
3. Enable more augmentation types
4. Increase curriculum schedule duration

### Issue: Slow Training

**Symptoms**: Augmentation takes too long

**Solutions**:

1. Disable blur: `apply_blur=False`
2. Increase DataLoader workers
3. Use smaller patch size
4. Consider caching augmented images

### Issue: Inconsistent Results

**Symptoms**: Results vary between runs

**Solutions**:

1. Set random seed: `seed=42`
2. Use fixed curriculum schedule
3. Ensure deterministic DataLoader: `worker_init_fn`

## References

1. **Masked Autoencoders Are Scalable Vision Learners**
   - Paper: <https://arxiv.org/abs/2111.06377>
   - Authors: Kaiming He et al.
   - Conference: CVPR 2022

2. **Curriculum Learning**
   - Paper: <https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf>
   - Authors: Bengio et al.
   - Conference: ICML 2009

3. **Data Augmentation for Visual Question Answering**
   - Paper: <https://aclanthology.org/W17-3529.pdf>
   - Authors: Li et al.
   - Conference: INLG 2017

## Related Documentation

- [Architecture Diagram](ARCHITECTURE_DIAGRAM.md) - System overview
- [Training Pipeline](../training/README.md) - Integration with training
- [Augmentation Factory](../augmentation/README.md) - Creating augmentors
- [Rule-Based Text Augmentation](RULE_BASED_AUGMENTATION.md) - Text augmentation

---

**Last Updated**: November 12, 2025
**Version**: 1.0.0
**Maintainers**: VQA Template Team
