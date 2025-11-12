# Smooth Curriculum Learning Scheduler

## Overview

The `CurriculumScheduler` provides fine-grained, continuous control over curriculum learning difficulty, similar to learning rate schedulers in deep learning. Unlike the discrete `CurriculumScheduler` which uses three fixed levels (EASY, MEDIUM, HARD), the smooth scheduler returns continuous difficulty values from 0.0 to 1.0.

## Why Smooth Scheduling?

**Traditional Discrete Approach:**
- Abrupt transitions between difficulty levels
- Limited control with only 3 levels
- May cause training instability at transitions

**Smooth Scheduling Benefits:**
- Gradual, continuous difficulty progression
- Fine-grained control over augmentation intensity
- Multiple scheduling strategies to match different training dynamics
- Better suited for modern deep learning workflows

## Scheduling Strategies

### 1. Linear
Uniform difficulty increase throughout training.

```python
scheduler = CurriculumScheduler(
    total_epochs=100,
    strategy='linear'
)
```

**Best for:** Stable, predictable progression; baseline experiments

**Progression:** `difficulty = epoch / total_epochs`

### 2. Cosine
Smooth S-curve progression (inspired by cosine annealing).

```python
scheduler = CurriculumScheduler(
    total_epochs=100,
    strategy='cosine'
)
```

**Best for:** Smooth transitions; mimics natural learning curves

**Progression:** `difficulty = (1 - cos(π * epoch / total)) / 2`

### 3. Exponential
Slow start with rapid increase toward the end.

```python
scheduler = CurriculumScheduler(
    total_epochs=100,
    strategy='exponential',
    gamma=0.05  # Controls growth rate
)
```

**Best for:** Extended easy training with gradual hardening

**Progression:** `difficulty = 1 - gamma^((total - epoch) / total)`

**Parameters:**
- `gamma`: Growth rate (lower = slower initial growth). Default: 0.1

### 4. Step
Step-wise increases at regular intervals.

```python
scheduler = CurriculumScheduler(
    total_epochs=100,
    strategy='step',
    step_size=20,  # Increase every 20 epochs
    gamma=0.25     # Increase by 0.25 each step
)
```

**Best for:** Controlled, predictable difficulty increases; debugging

**Parameters:**
- `step_size`: Epochs between increases. Default: `total_epochs // 3`
- `gamma`: Difficulty increase per step. Default: 0.33

### 5. Polynomial
Power-law progression.

```python
scheduler = CurriculumScheduler(
    total_epochs=100,
    strategy='polynomial',
    power=2.0  # Quadratic progression
)
```

**Best for:** Customizable progression curves

**Progression:** `difficulty = (epoch / total)^power`

**Parameters:**
- `power`: Exponent (higher = slower initial growth). Default: 2.0

## Advanced Features

### Custom Difficulty Range

Control the min/max difficulty values:

```python
scheduler = CurriculumScheduler(
    total_epochs=100,
    strategy='linear',
    min_difficulty=0.2,  # Start at 20% difficulty
    max_difficulty=0.8   # Cap at 80% difficulty
)
```

**Use cases:**
- Avoid trivial samples: `min_difficulty=0.2`
- Prevent overly aggressive augmentation: `max_difficulty=0.8`
- Fine-tune for specific datasets

### Warmup Period

Keep difficulty low for initial epochs:

```python
scheduler = CurriculumScheduler(
    total_epochs=100,
    strategy='cosine',
    warmup_epochs=10  # First 10 epochs at min_difficulty
)
```

**Benefits:**
- Stabilize early training
- Allow model to learn basic patterns first
- Reduce variance in initial gradients

## Practical Usage

### Image Augmentation Example

```python
from augmentation import CurriculumScheduler

# Initialize scheduler
scheduler = CurriculumScheduler(
    total_epochs=100,
    strategy='cosine',
    warmup_epochs=5
)

# Training loop
for epoch in range(100):
    difficulty = scheduler.get_difficulty(epoch)
    
    # Use difficulty to control augmentation parameters
    mask_ratio = 0.1 + difficulty * 0.4  # 10% to 50%
    rotation_angle = difficulty * 30      # 0° to 30°
    noise_std = 0.01 + difficulty * 0.09  # 0.01 to 0.1
    
    # Apply augmentation with computed parameters
    augmented_images = augment_images(
        images,
        mask_ratio=mask_ratio,
        rotation_angle=rotation_angle,
        noise_std=noise_std
    )
    
    # Train model
    train_step(augmented_images)
```

### Text Augmentation Example

```python
for epoch in range(100):
    difficulty = scheduler.get_difficulty(epoch)
    
    # Control text augmentation intensity
    synonym_ratio = 0.1 + difficulty * 0.3    # 10% to 40%
    word_drop_prob = difficulty * 0.2          # 0% to 20%
    paraphrase_strength = difficulty           # 0.0 to 1.0
    
    augmented_text = augment_text(
        text,
        synonym_ratio=synonym_ratio,
        word_drop_prob=word_drop_prob,
        paraphrase_strength=paraphrase_strength
    )
```

### Integration with Augmentation Classes

If your augmentation class accepts difficulty as a float:

```python
from augmentation import MaskedImageAugmentation, CurriculumScheduler

scheduler = CurriculumScheduler(total_epochs=50, strategy='cosine')
augmenter = MaskedImageAugmentation()

for epoch in range(50):
    difficulty = scheduler.get_difficulty(epoch)
    
    # Update augmenter difficulty
    augmenter.difficulty_value = difficulty
    
    # Apply augmentation
    for batch in dataloader:
        augmented = augmenter.augment(batch['image'], difficulty=difficulty)
```

## Comparison with Discrete Scheduler

### When to use Discrete Scheduler

```python
from augmentation import CurriculumScheduler

scheduler = CurriculumScheduler(
    total_epochs=30,
    easy_epochs=10,
    medium_epochs=10,
    hard_epochs=10
)
```

**Use when:**
- Simple, interpretable curriculum needed
- Working with discrete augmentation strategies
- Quick prototyping
- Need backward compatibility

**Can enable smooth mode:**
```python
scheduler = CurriculumScheduler(
    total_epochs=30,
    smooth_transition=True  # Returns 0.0, 0.5, 1.0
)
```

### When to use Smooth Scheduler

```python
from augmentation import CurriculumScheduler

scheduler = CurriculumScheduler(
    total_epochs=100,
    strategy='cosine',
    warmup_epochs=10
)
```

**Use when:**
- Need fine-grained difficulty control
- Want to avoid abrupt transitions
- Experimenting with different progression curves
- Mimicking LR scheduler patterns
- Working with continuous augmentation parameters

## Visualization

Run the example script to visualize different strategies:

```bash
cd examples
python curriculum_scheduler_usage.py
```

This generates comparison plots showing:
- All scheduling strategies side-by-side
- Warmup functionality
- Custom difficulty ranges
- Practical augmentation parameters over time

## API Reference

### CurriculumScheduler

```python
CurriculumScheduler(
    total_epochs: int,
    strategy: Literal['linear', 'cosine', 'exponential', 'step', 'polynomial'] = 'linear',
    min_difficulty: float = 0.0,
    max_difficulty: float = 1.0,
    warmup_epochs: int = 0,
    **kwargs
)
```

**Methods:**

- `get_difficulty(epoch: int) -> float`: Get difficulty value for epoch
- `get_schedule_info() -> dict`: Get configuration and sample values

**Strategy-specific kwargs:**

- **exponential**: `gamma=0.1` (growth rate)
- **step**: `step_size=total_epochs//3`, `gamma=0.33` (step increase)
- **polynomial**: `power=2.0` (exponent)

### CurriculumScheduler (Enhanced)

```python
CurriculumScheduler(
    total_epochs: int,
    easy_epochs: Optional[int] = None,
    medium_epochs: Optional[int] = None,
    hard_epochs: Optional[int] = None,
    smooth_transition: bool = False
)
```

**Methods:**

- `get_difficulty_for_epoch(epoch: int) -> Union[DifficultyLevel, float]`
- `get_schedule_info() -> dict`

## Best Practices

1. **Start Simple**: Begin with linear or cosine strategies
2. **Use Warmup**: For stability, add 5-10% warmup epochs
3. **Custom Ranges**: Adjust min/max based on dataset characteristics
4. **Log Difficulty**: Track difficulty values in training logs
5. **Visualize**: Plot schedules before training to verify behavior
6. **Experiment**: Try different strategies; no one-size-fits-all

## Example Configurations

### Quick Start (Stable)
```python
CurriculumScheduler(total_epochs=100, strategy='linear')
```

### Recommended (Smooth)
```python
CurriculumScheduler(
    total_epochs=100,
    strategy='cosine',
    warmup_epochs=10
)
```

### Conservative (Gentle)
```python
CurriculumScheduler(
    total_epochs=100,
    strategy='exponential',
    gamma=0.03,
    warmup_epochs=20
)
```

### Aggressive (Fast)
```python
CurriculumScheduler(
    total_epochs=100,
    strategy='polynomial',
    power=3.0
)
```

### Custom Range
```python
CurriculumScheduler(
    total_epochs=100,
    strategy='cosine',
    min_difficulty=0.3,
    max_difficulty=0.7,
    warmup_epochs=10
)
```

## Migration Guide

### From Discrete to Smooth

**Before:**
```python
scheduler = CurriculumScheduler(total_epochs=30)
difficulty = scheduler.get_difficulty(epoch)  # DifficultyLevel.EASY/MEDIUM/HARD

# Use with if-else
if difficulty == DifficultyLevel.EASY:
    mask_ratio = 0.1
elif difficulty == DifficultyLevel.MEDIUM:
    mask_ratio = 0.3
else:
    mask_ratio = 0.5
```

**After:**
```python
scheduler = CurriculumScheduler(total_epochs=30, strategy='cosine')
difficulty = scheduler.get_difficulty(epoch)  # Float: 0.0 to 1.0

# Interpolate parameters
mask_ratio = 0.1 + difficulty * 0.4  # 0.1 to 0.5
```

## See Also

- [RULE_BASED_AUGMENTATION.md](RULE_BASED_AUGMENTATION.md) - Text augmentation strategies
- [MASKING_IMAGE_AUGMENTATION.md](MASKING_IMAGE_AUGMENTATION.md) - Image augmentation
- `examples/curriculum_scheduler_usage.py` - Complete examples
