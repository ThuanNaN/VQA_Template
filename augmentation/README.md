# Augmentation Module for VQA

This module provides comprehensive data augmentation for Visual Question Answering (VQA) tasks, including both **visual augmentation** (image-based) and **textual augmentation** (question/text-based). The module is designed with flexibility, extensibility, and curriculum learning in mind.

## Overview

The augmentation module supports:

- 🖼️ **Visual Augmentation**: MAE-inspired patch masking, color transformations, geometric transforms
- 📝 **Textual Augmentation**: Rule-based paraphrasing for Vietnamese questions
- 📚 **Curriculum Learning**: Progressive difficulty increase during training
- 🏭 **Factory Pattern**: Easy creation and configuration of augmentation strategies
- 🔧 **Extensible Design**: Simple addition of custom augmentation methods

## Module Structure

```
augmentation/
├── __init__.py           # Main exports and public API
├── base.py               # Base classes and interfaces
├── factory.py            # Factory for creating augmentors
├── scheduler.py          # Curriculum learning scheduler
├── visual/              
│   ├── __init__.py
│   └── mask.py          # MAE-inspired image augmentation
└── textual/
    ├── __init__.py
    └── rule_based.py    # Rule-based text augmentation
```

## Features

### Visual Augmentation

- **MAE-inspired Patch Masking**: Randomly mask image patches (15%, 50%, or 75% based on difficulty)
- **Color Transformations**: Adaptive color jittering, brightness, and contrast adjustments
- **Blur Operations**: Gaussian blur with varying intensities
- **Geometric Transformations**: Random cropping and horizontal flipping
- **Reproducibility**: Seeded random number generator for consistent results

### Textual Augmentation

- **Question Word Variations**: Vietnamese question word paraphrasing (gì ↔ cái gì, đâu ↔ nơi nào)
- **Synonym Replacement**: Context-aware synonym substitution
- **Demonstrative Pronoun Variations**: Variations for này/đây/nầy and kia/đó/ấy
- **Combined Strategies**: Multiple transformations for richer variations

### General Features

- **Curriculum Learning**: Progressive difficulty increase during training (EASY → MEDIUM → HARD)
- **Factory Pattern**: Unified interface for creating different augmentation types
- **Flexible Configuration**: Toggle individual augmentation techniques on/off
- **Extensible Architecture**: Easy to add custom augmentation strategies

## Quick Start

### Using the Factory Pattern (Recommended)

```python
from augmentation import AugmentationFactory

# Create visual augmentation
factory = AugmentationFactory()
image_augmentor = factory.create_image_augmentation(
    augmentation_type='masked',
    difficulty='medium',
    patch_size=16,
    seed=42
)

# Create text augmentation
text_augmentor = factory.create_text_augmentation(
    augmentation_type='rule-based',
    seed=42
)

# Apply augmentations
augmented_image = image_augmentor.augment(image)
augmented_questions = text_augmentor.augment(question, num_augmentations=3)
```

### Visual Augmentation - Basic Usage

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

### Textual Augmentation - Basic Usage

```python
from augmentation import RuleBasedTextAugmentation

# Create text augmentor
augmentor = RuleBasedTextAugmentation(seed=42)

# Augment a Vietnamese question
question = "Cái gì trong ảnh này?"
augmented_questions = augmentor.augment(question, num_augmentations=3)

# Output:
# ["Thứ gì trong ảnh này?", "Cái gì trong ảnh đây?", "Điều gì trong ảnh nầy?"]
```

### Curriculum Learning

```python
from augmentation import CurriculumLearningScheduler, MaskedImageAugmentation

# Create scheduler for 30 epochs
scheduler = CurriculumLearningScheduler(total_epochs=30)

# Training loop
for epoch in range(30):
    # Get difficulty for current epoch
    difficulty = scheduler.get_difficulty_for_epoch(epoch)
    augmentor = MaskedImageAugmentation(difficulty=difficulty, seed=42)
    
    # Train with appropriate difficulty
    # ...
```

### Integration with VQA Dataset

```python
from augmentation import AugmentationFactory, DifficultyLevel
from torch.utils.data import Dataset

class AugmentedVQADataset(Dataset):
    def __init__(self, base_dataset, difficulty=DifficultyLevel.EASY, 
                 use_text_aug=True, use_image_aug=True):
        self.base_dataset = base_dataset
        factory = AugmentationFactory()
        
        # Create augmentors
        self.image_augmentor = None
        self.text_augmentor = None
        
        if use_image_aug:
            self.image_augmentor = factory.create_image_augmentation(
                augmentation_type='masked',
                difficulty=difficulty
            )
        
        if use_text_aug:
            self.text_augmentor = factory.create_text_augmentation(
                augmentation_type='rule-based'
            )
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        # Apply image augmentation
        if self.image_augmentor:
            item['image'] = self.image_augmentor.augment(item['image'])
        
        # Apply text augmentation (randomly)
        if self.text_augmentor and random.random() > 0.5:
            augmented = self.text_augmentor.augment(item['question'], num_augmentations=1)
            item['question'] = augmented[0] if augmented else item['question']
        
        return item
```

## Augmentation Types

### Visual Augmentation: Masked Image Augmentation

Inspired by [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (CVPR 2022), this augmentation randomly masks image patches to improve model robustness.

#### Difficulty Levels

##### EASY (15% masking)

- Minimal augmentation for early training
- Slight color jittering (±10%)
- Small brightness/contrast adjustments (0.9-1.1x)
- Light blur (radius: 0.1-0.3)
- No horizontal flipping
- Minimal cropping (95-100%)

**Use case**: Initial epochs when model is learning basic features

##### MEDIUM (50% masking)

- Moderate augmentation for intermediate training
- Medium color jittering (±30%)
- Moderate brightness/contrast adjustments (0.7-1.3x)
- Medium blur (radius: 0.5-1.5)
- Random horizontal flipping enabled
- Medium cropping (80-100%)

**Use case**: Mid-training when model needs more challenging samples

##### HARD (75% masking)

- Aggressive augmentation for final training phases
- Strong color jittering (±50%)
- Large brightness/contrast adjustments (0.5-1.5x)
- Strong blur (radius: 1.0-3.0)
- Random horizontal flipping enabled
- Aggressive cropping (70-100%)

**Use case**: Later epochs to improve model robustness and generalization

## Curriculum Learning Schedules

We provide two types of curriculum learning schedulers:

### 1. Discrete Scheduler (Traditional)

The traditional `CurriculumLearningScheduler` uses three discrete difficulty levels.

**Default schedule for 30 epochs:**

| Epochs | Difficulty | Mask Ratio | Purpose |
|--------|-----------|-----------|---------|
| 0-8 (30%) | EASY | 15% | Learn basic features |
| 9-17 (30%) | MEDIUM | 50% | Intermediate robustness |
| 18-29 (40%) | HARD | 75% | Strong generalization |

**Custom Schedule:**

```python
# Custom distribution: longer easy period
scheduler = CurriculumLearningScheduler(
    total_epochs=50,
    easy_epochs=20,    # 40%
    medium_epochs=20,  # 40%
    hard_epochs=10     # 20%
)

# Enable smooth transition mode (returns 0.0, 0.5, 1.0)
scheduler = CurriculumLearningScheduler(
    total_epochs=30,
    smooth_transition=True
)
```

### 2. Smooth Scheduler (Recommended) 🆕

The new `CurriculumScheduler` provides continuous difficulty values (0.0 to 1.0) for fine-grained control.

**Five scheduling strategies:**

```python
from augmentation import CurriculumScheduler

# Linear progression (uniform increase)
scheduler = CurriculumScheduler(total_epochs=100, strategy='linear')

# Cosine annealing (smooth S-curve) - RECOMMENDED
scheduler = CurriculumScheduler(total_epochs=100, strategy='cosine')

# Exponential (slow start, rapid increase)
scheduler = CurriculumScheduler(total_epochs=100, strategy='exponential', gamma=0.05)

# Step-wise increases
scheduler = CurriculumScheduler(total_epochs=100, strategy='step', step_size=20)

# Polynomial progression
scheduler = CurriculumScheduler(total_epochs=100, strategy='polynomial', power=2.5)
```

**Advanced features:**

```python
# With warmup period
scheduler = CurriculumScheduler(
    total_epochs=100,
    strategy='cosine',
    warmup_epochs=10,          # First 10 epochs at minimum
    min_difficulty=0.2,        # Start at 20%
    max_difficulty=0.8         # Cap at 80%
)

# Use in training loop
for epoch in range(100):
    difficulty = scheduler.get_difficulty(epoch)  # Returns 0.0 to 1.0
    
    # Interpolate augmentation parameters
    mask_ratio = 0.1 + difficulty * 0.4    # 10% to 50%
    synonym_ratio = 0.1 + difficulty * 0.3  # 10% to 40%
```

**Why use Smooth Scheduler?**
- ✓ Continuous difficulty values (not just 3 levels)
- ✓ Smooth transitions (no abrupt jumps)
- ✓ Multiple strategies (linear, cosine, exponential, step, polynomial)
- ✓ Warmup support for training stability
- ✓ Custom difficulty ranges
- ✓ Similar to learning rate schedulers

📖 **See full documentation:** [docs/SMOOTH_CURRICULUM_SCHEDULER.md](../docs/SMOOTH_CURRICULUM_SCHEDULER.md)

🎯 **Quick reference:** [SCHEDULER_QUICK_REFERENCE.md](../SCHEDULER_QUICK_REFERENCE.md)

### Textual Augmentation: Rule-Based Text Augmentation

Inspired by "Data Augmentation for Visual Question Answering" (ACL 2017), this module provides Vietnamese-specific text augmentation through linguistic rules.

#### Augmentation Strategies

1. **Question Word Variations**
   - `gì` ↔ `cái gì`, `thứ gì`, `điều gì`
   - `đâu` ↔ `nơi nào`, `chỗ nào`, `ở đâu`
   - `khi nào` ↔ `lúc nào`, `bao giờ`

2. **Synonym Replacement**
   - Colors: `đỏ` ↔ `đỏ thẫm`, `son`
   - Verbs: `làm` ↔ `thực hiện`, `tiến hành`
   - Adjectives: `lớn` ↔ `to`, `rộng`

3. **Demonstrative Pronouns**
   - `này` ↔ `đây`, `nầy`
   - `kia` ↔ `đó`, `ấy`

#### Usage Example

```python
from augmentation import RuleBasedTextAugmentation

augmentor = RuleBasedTextAugmentation(seed=42)

# Single question augmentation
question = "Cái gì trong ảnh này?"
augmented = augmentor.augment(question, num_augmentations=3)
# Output: ["Thứ gì trong ảnh này?", "Cái gì trong ảnh đây?", ...]

# Batch augmentation
questions = ["Người này làm gì?", "Màu gì của chiếc xe?"]
all_augmented = [augmentor.augment(q, num_augmentations=2) for q in questions]
```

For more details on Vietnamese text augmentation, see [docs/RULE_BASED_AUGMENTATION.md](../docs/RULE_BASED_AUGMENTATION.md).

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

### AugmentationFactory

Factory class for creating augmentation instances.

```python
factory = AugmentationFactory()

# Create image augmentation
image_aug = factory.create_image_augmentation(
    augmentation_type='masked',  # or 'none'
    difficulty='medium',
    patch_size=16,
    seed=42
)

# Create text augmentation
text_aug = factory.create_text_augmentation(
    augmentation_type='rule-based',  # or 'none'
    seed=42
)

# List available types
print(factory.list_available_types())
# {'image': ['masked', 'none'], 'text': ['rule-based', 'none']}
```

### MaskedImageAugmentation

Visual augmentation with patch masking and transformations.

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

### RuleBasedTextAugmentation

Vietnamese text augmentation using linguistic rules.

```python
RuleBasedTextAugmentation(seed: Optional[int] = None)
```

**Parameters:**

- `seed`: Random seed for reproducibility

**Methods:**

- `augment(text, num_augmentations=1)`: Generate augmented versions of text
  - Returns: List of augmented text strings
- `get_augmentation_info()`: Get current configuration

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

## Examples

Complete examples are available in the `examples/` directory:

**Visual Augmentation:**

1. `image_aug_usage.py` - Basic usage and feature demonstration
2. Dataset integration examples in VQA dataset files

**Textual Augmentation:**

1. `text_aug_usage.py` - Vietnamese question augmentation
2. Dataset integration examples

**Combined Usage:**

```python
from augmentation import AugmentationFactory, CurriculumLearningScheduler

# Setup
factory = AugmentationFactory()
scheduler = CurriculumLearningScheduler(total_epochs=30)

# Training loop with both augmentations
for epoch in range(30):
    difficulty = scheduler.get_difficulty_for_epoch(epoch)
    
    # Create augmentors for current epoch
    image_aug = factory.create_image_augmentation('masked', difficulty=difficulty)
    text_aug = factory.create_text_augmentation('rule-based')
    
    for batch in dataloader:
        # Apply augmentations
        batch['images'] = [image_aug.augment(img) for img in batch['images']]
        # Optionally augment questions
        if random.random() > 0.5:
            batch['questions'] = [
                text_aug.augment(q, num_augmentations=1)[0] 
                for q in batch['questions']
            ]
        # Train...
```

Run examples:

```bash
cd examples
python image_aug_usage.py
python text_aug_usage.py
```

## Research Background

This implementation is inspired by multiple research works:

### Masked Autoencoders Are Scalable Vision Learners (He et al., CVPR 2022)

- Paper: <https://arxiv.org/abs/2111.06377>
- Key insight: Random masking of 75% of image patches forces models to learn robust representations
- Our implementation: Provides configurable masking ratios for curriculum learning

### Data Augmentation for Visual Question Answering (ACL 2017)

- Concept: Use linguistic rules to generate paraphrased questions
- Our approach: Vietnamese-specific rule-based augmentation with question word variations and synonyms
- Benefit: Increased training data diversity without manual annotation

### Curriculum Learning (Bengio et al., ICML 2009)

- Concept: Train models on progressively harder examples
- Our approach: Easy (15% masking) → Medium (50%) → Hard (75%)
- Benefit: Better convergence and improved generalization

## Best Practices

### For Visual Augmentation

1. **Start with Easy**: Begin training with EASY difficulty to learn basic features
2. **Monitor Performance**: Track accuracy at each difficulty level transition
3. **Adjust Schedule**: Customize epoch distribution based on dataset complexity
4. **Validation**: Always validate on non-augmented images for fair evaluation

### For Textual Augmentation

1. **Selective Application**: Don't augment all questions - use ~50% probability
2. **Validate Semantics**: Ensure augmented questions preserve meaning
3. **Monitor Quality**: Check samples to verify augmentation quality
4. **Language-Specific**: Current implementation is Vietnamese-specific

### General Guidelines

1. **Combine Wisely**: Use both visual and textual augmentation for best results
2. **Experiment**: Try different combinations and augmentation rates
3. **Logging**: Track augmentation parameters in your experiment logs
4. **Reproducibility**: Always set seeds for reproducible experiments
5. **Dataset Size**: More augmentation helps smaller datasets, less for larger ones

## Related Documentation

- [docs/RULE_BASED_AUGMENTATION.md](../docs/RULE_BASED_AUGMENTATION.md) - Detailed Vietnamese text augmentation guide
- [examples/image_aug_usage.py](../examples/image_aug_usage.py) - Visual augmentation examples
- [examples/text_aug_usage.py](../examples/text_aug_usage.py) - Textual augmentation examples
- [training/README.md](../training/README.md) - Training pipeline integration

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
```

**For Visual Augmentation (MAE-inspired):**

```bibtex
@inproceedings{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16000--16009},
  year={2022}
}
```

**For Textual Augmentation (Rule-based):**

```bibtex
@inproceedings{li2017data,
  title={Data Augmentation for Visual Question Answering},
  author={Li, Qing and Fu, Jianlong and Yu, Dongfei and Mei, Tao and Luo, Jiebo},
  booktitle={Proceedings of the 10th International Conference on Natural Language Generation},
  pages={198--202},
  year={2017}
}
```

## License

This augmentation framework is part of the VQA_Template project and follows the same license.
