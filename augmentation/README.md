# Image Augmentation with Curriculum Learning

This module provides image augmentation for Visual Question Answering (VQA) tasks, inspired by [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (CVPR 2022), with support for Curriculum Learning to guide model training from easy to hard samples.

## Features

- **MAE-inspired Patch Masking**: Randomly mask image patches (15%, 50%, or 75% based on difficulty)
- **Color Transformations**: Adaptive color jittering, brightness, and contrast adjustments
- **Blur Operations**: Gaussian blur with varying intensities
- **Geometric Transformations**: Random cropping and horizontal flipping
- **Curriculum Learning**: Progressive difficulty increase during training (EASY → MEDIUM → HARD)
- **Reproducibility**: Seeded random number generator for consistent results
- **Flexible Configuration**: Toggle individual augmentation techniques on/off

## Quick Start

### Basic Usage

```python
from augmentation import MaskedImageAugmentation, DifficultyLevel
from PIL import Image

# Load an image
image = Image.open('path/to/image.jpg')

# Create augmentor with MEDIUM difficulty
augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.MEDIUM)

# Apply augmentation
augmented_image = augmentor.augment(image)
```

### Curriculum Learning

```python
from augmentation import CurriculumLearningScheduler, create_augmentor_for_epoch

# Create scheduler for 30 epochs
scheduler = CurriculumLearningScheduler(total_epochs=30)

# Training loop
for epoch in range(30):
    # Get augmentor for current epoch
    augmentor = create_augmentor_for_epoch(epoch, scheduler, seed=42)
    
    # Train with appropriate difficulty
    # ...
```

### Integration with VQA Dataset

```python
from augmentation import MaskedImageAugmentation, DifficultyLevel
from torch.utils.data import Dataset

class AugmentedVQADataset(Dataset):
    def __init__(self, base_dataset, difficulty=DifficultyLevel.EASY):
        self.base_dataset = base_dataset
        self.augmentor = MaskedImageAugmentation(difficulty=difficulty)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        # Apply augmentation to PIL image before converting to tensor
        item['image'] = self.augmentor.augment(item['image'])
        return item
```

## Difficulty Levels

### EASY (15% masking)
- Minimal augmentation for early training
- Slight color jittering (±10%)
- Small brightness/contrast adjustments (0.9-1.1x)
- Light blur (radius: 0.1-0.3)
- No horizontal flipping
- Minimal cropping (95-100%)

**Use case**: Initial epochs when model is learning basic features

### MEDIUM (50% masking)
- Moderate augmentation for intermediate training
- Medium color jittering (±30%)
- Moderate brightness/contrast adjustments (0.7-1.3x)
- Medium blur (radius: 0.5-1.5)
- Random horizontal flipping enabled
- Medium cropping (80-100%)

**Use case**: Mid-training when model needs more challenging samples

### HARD (75% masking)
- Aggressive augmentation for final training phases
- Strong color jittering (±50%)
- Large brightness/contrast adjustments (0.5-1.5x)
- Strong blur (radius: 1.0-3.0)
- Random horizontal flipping enabled
- Aggressive cropping (70-100%)

**Use case**: Later epochs to improve model robustness and generalization

## Curriculum Learning Schedule

The default schedule for 30 epochs:

| Epochs | Difficulty | Mask Ratio | Purpose |
|--------|-----------|-----------|---------|
| 0-8 (30%) | EASY | 15% | Learn basic features |
| 9-17 (30%) | MEDIUM | 50% | Intermediate robustness |
| 18-29 (40%) | HARD | 75% | Strong generalization |

### Custom Schedule

```python
# Custom distribution: longer easy period
scheduler = CurriculumLearningScheduler(
    total_epochs=50,
    easy_epochs=20,    # 40%
    medium_epochs=20,  # 40%
    hard_epochs=10     # 20%
)
```

## Selective Augmentation

Apply only specific augmentation techniques:

```python
augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.MEDIUM)

# Only apply masking
masked = augmentor.augment(
    image,
    apply_masking=True,
    apply_color_jitter=False,
    apply_blur=False,
    apply_brightness=False,
    apply_contrast=False
)

# Only color transformations
color_aug = augmentor.augment(
    image,
    apply_masking=False,
    apply_color_jitter=True,
    apply_brightness=True,
    apply_contrast=True
)
```

## API Reference

### MaskedImageAugmentation

```python
MaskedImageAugmentation(
    difficulty: Union[DifficultyLevel, str] = DifficultyLevel.EASY,
    patch_size: int = 16,
    mask_ratio: Optional[float] = None,
    seed: Optional[int] = None
)
```

**Parameters:**
- `difficulty`: Augmentation difficulty level (EASY, MEDIUM, or HARD)
- `patch_size`: Size of patches for masking (default: 16)
- `mask_ratio`: Custom mask ratio (overrides difficulty default)
- `seed`: Random seed for reproducibility

**Methods:**

- `augment(image, **kwargs)`: Apply augmentation to image
- `get_augmentation_info()`: Get current configuration parameters

### CurriculumLearningScheduler

```python
CurriculumLearningScheduler(
    total_epochs: int,
    easy_epochs: Optional[int] = None,
    medium_epochs: Optional[int] = None,
    hard_epochs: Optional[int] = None
)
```

**Parameters:**
- `total_epochs`: Total number of training epochs
- `easy_epochs`: Number of EASY epochs (default: 30% of total)
- `medium_epochs`: Number of MEDIUM epochs (default: 30% of total)
- `hard_epochs`: Number of HARD epochs (default: 40% of total)

**Methods:**

- `get_difficulty_for_epoch(epoch)`: Get difficulty level for given epoch
- `get_schedule_info()`: Get schedule information dictionary

### Utility Functions

```python
create_augmentor_for_epoch(
    epoch: int,
    scheduler: CurriculumLearningScheduler,
    patch_size: int = 16,
    seed: Optional[int] = None
) -> MaskedImageAugmentation
```

Creates an augmentor configured for the current epoch based on the curriculum schedule.

## Examples

See the `examples/` directory for complete examples:

1. **`image_augmentation_usage.py`**: Basic usage and feature demonstration
2. **`image_augmentation_integration.py`**: Integration with VQA datasets and training loops
3. **`test_image_augmentation.py`**: Validation tests for all features

Run examples:

```bash
cd examples
python image_augmentation_usage.py
python image_augmentation_integration.py
python test_image_augmentation.py
```

## Research Background

This implementation is inspired by:

**Masked Autoencoders Are Scalable Vision Learners** (He et al., CVPR 2022)
- Paper: https://arxiv.org/abs/2111.06377
- Key insight: Random masking of 75% of image patches forces models to learn robust representations
- Our implementation: Provides configurable masking ratios for curriculum learning

**Curriculum Learning** (Bengio et al., ICML 2009)
- Concept: Train models on progressively harder examples
- Our approach: Easy (15% masking) → Medium (50%) → Hard (75%)
- Benefit: Better convergence and improved generalization

## Tips for Best Results

1. **Start with Easy**: Begin training with EASY difficulty to learn basic features
2. **Monitor Performance**: Track accuracy at each difficulty level transition
3. **Adjust Schedule**: Customize epoch distribution based on dataset complexity
4. **Validation**: Always validate on non-augmented images for fair evaluation
5. **Experiment**: Try different masking ratios and augmentation combinations
6. **Logging**: Track augmentation parameters in your experiment logs

## Citation

If you use this augmentation framework in your research, please cite:

```bibtex
@misc{vqa_template_2024,
  title={Vietnamese VQA Template: Datasets, Analysis, and Tools},
  author={VQA Template Team},
  year={2024},
  howpublished={\url{https://github.com/ThuanNaN/VQA_Template}},
  note={Framework and comprehensive survey for Vietnamese Visual Question Answering}
}

@inproceedings{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16000--16009},
  year={2022}
}
```

## License

This augmentation framework is part of the VQA_Template project and follows the same license.
