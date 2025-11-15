# Dynamic Multi-Input Augmentation

This feature enables **dynamic input handling** for both textual and visual data, supporting both **single and multiple inputs** with automatic aggregation.

## Overview

The VQA model can now process:
- **Single or multiple text inputs** (e.g., paraphrases)
- **Single or multiple image inputs** (e.g., multi-view augmentations)
- **Automatic shape detection** - no manual configuration needed
- **Flexible aggregation** - mean, sum, max, or first

## How It Works

### Architecture Flow

```
Input → Augmentation → Dataset → Encoder → Aggregation → Classifier
```

1. **Augmentation** returns either:
   - Single output: `"text"` or `PIL.Image`
   - Multiple outputs: `["text1", "text2", "text3"]` or `[img1, img2, img3]`

2. **Dataset** detects output type and tokenizes/processes accordingly:
   - Single: `[seq_len]` or `[C, H, W]`
   - Multiple: `[N, seq_len]` or `[N, C, H, W]`

3. **Encoder** detects input shape:
   - If 3D/4D (single): Normal encoding
   - If 4D/5D (multiple): Batch encoding + aggregation

4. **Aggregation** combines multiple embeddings:
   - `mean`: Average embeddings (default)
   - `sum`: Sum embeddings
   - `max`: Max pooling across embeddings
   - `first`: Use only first (disables ensemble)

## Usage Examples

### Example 1: Simple Paraphrase Augmentation

```python
from augmentation import AugmentationFactory

# Create paraphrase augmentation
text_aug = AugmentationFactory.create_text_augmentation(
    augmentation_type='simple-paraphrase',
    difficulty=0.5,
    num_paraphrases=3  # Returns 3 paraphrases
)

# Test it
question = "Có bao nhiêu con chó?"
paraphrases = text_aug.augment(question)
# Returns: ["Có bao nhiêu con chó?", "Có mấy con chó?", "Bao nhiêu con chó?"]
```

### Example 2: Multi-View Image Augmentation

```python
from augmentation import AugmentationFactory
from PIL import Image

# Create multi-view augmentation
image_aug = AugmentationFactory.create_image_augmentation(
    augmentation_type='multi-view',
    difficulty=0.5,
    num_views=4  # Returns 4 augmented views
)

# Test it
image = Image.open("image.jpg")
views = image_aug.augment(image)
# Returns: [original_img, rotated_img, brightness_img, contrast_img]
```

### Example 3: Full Training Pipeline

```python
from training import ExperimentConfig, ModelConfig, AugmentationConfig

config = ExperimentConfig(
    model=ModelConfig(
        text_aggregation='mean',  # Aggregate paraphrases
        vis_aggregation='max',    # Aggregate multi-view
    ),
    augmentation=AugmentationConfig(
        text_augmentation_type='simple-paraphrase',
        image_augmentation_type='multi-view',
    )
)
```

### Example 4: CLI Training

```bash
python train.py \
  --dataset_name vivqa \
  --enable_augmentation \
  --text_augmentation_type simple-paraphrase \
  --image_augmentation_type multi-view \
  --text_aggregation mean \
  --vis_aggregation max \
  --epochs 30
```

## Available Augmentations

### Text Augmentations (Return Multiple)

| Type | Description | Returns |
|------|-------------|---------|
| `simple-paraphrase` | Rule-based Vietnamese paraphrasing | List of paraphrased texts |
| `paraphrase` | LLM-based paraphrase generation | List of paraphrased texts |

### Image Augmentations (Return Multiple)

| Type | Description | Returns |
|------|-------------|---------|
| `multi-view` | Multiple augmented views (rotation, brightness, etc.) | List of augmented images |
| `crop-multi-view` | Multiple crops (center, corners, random) | List of cropped images |

### Traditional Augmentations (Return Single)

| Type | Description | Returns |
|------|-------------|---------|
| `rule-based` | Synonym replacement, word swap | Single augmented text |
| `masked` | Patch masking | Single masked image |

## Aggregation Methods

| Method | Formula | Best For |
|--------|---------|----------|
| `mean` | `avg(e1, e2, ..., eN)` | Balanced ensemble (default) |
| `sum` | `e1 + e2 + ... + eN` | Preserving total information |
| `max` | `max(e1, e2, ..., eN)` | Capturing strongest features |
| `first` | `e1` | Disabling ensemble (baseline) |

## Configuration

### Programmatic Configuration

```python
from training.config import ModelConfig

config = ModelConfig(
    vis_model_name='google/vit-base-patch16-224',
    text_model_name='vinai/bartpho-syllable-base',
    text_aggregation='mean',  # Options: mean, sum, max, first
    vis_aggregation='mean',   # Options: mean, sum, max, first
)
```

### CLI Arguments

```bash
--text_aggregation mean     # Text embedding aggregation method
--vis_aggregation max       # Image embedding aggregation method
```

## Creating Custom Multi-Input Augmentations

### Custom Text Augmentation

```python
from augmentation.base import BaseTextAugmentation
from typing import List

class MyParaphraseAugmentation(BaseTextAugmentation):
    def __init__(self, difficulty: float = 0.5, num_outputs: int = 3):
        super().__init__(difficulty)
        self.num_outputs = num_outputs
    
    def _configure_parameters(self):
        pass
    
    def augment(self, text: str) -> List[str]:
        """Return list of paraphrases"""
        paraphrases = [text]  # Include original
        # Add your paraphrase logic here
        for _ in range(self.num_outputs - 1):
            paraphrase = self.my_paraphrase_method(text)
            paraphrases.append(paraphrase)
        return paraphrases
```

### Custom Image Augmentation

```python
from augmentation.base import BaseImageAugmentation
from typing import List
from PIL import Image

class MyMultiViewAugmentation(BaseImageAugmentation):
    def __init__(self, difficulty: float = 0.5, num_views: int = 3):
        super().__init__(difficulty)
        self.num_views = num_views
    
    def _configure_parameters(self):
        pass
    
    def augment(self, image: Image.Image) -> List[Image.Image]:
        """Return list of augmented views"""
        views = [image.copy()]  # Include original
        # Add your augmentation logic here
        for _ in range(self.num_views - 1):
            view = self.my_augment_method(image)
            views.append(view)
        return views
```

## Performance Considerations

### Memory Usage

- **Single input**: `batch_size` samples
- **Multiple inputs**: `batch_size × num_inputs` effective samples
- Example: batch=32, 3 paraphrases, 3 views = 32 × 3 × 3 = **288 effective samples**

**Recommendation**: Reduce `batch_size` when using multi-input augmentation

### Training Speed

- **Encoding overhead**: `num_texts × num_images` forward passes per sample
- **Aggregation**: Negligible overhead (mean/sum/max pooling)

**Recommendation**: Use offline paraphrase generation for faster training

### GPU Memory

Approximate GPU memory increase:
- 3 paraphrases: ~3x text encoder memory
- 3 image views: ~3x vision encoder memory
- Total: ~9x for full multi-input (3 texts × 3 views)

**Recommendation**: 
- Use gradient checkpointing: `--gradient_checkpointing`
- Use mixed precision: `--fp16`
- Reduce hidden_size if needed

## Experiments

Recommended experiment configurations:

### Baseline
```bash
python train.py --dataset_name vivqa --epochs 30
```

### Text Ensemble Only
```bash
python train.py --dataset_name vivqa \
  --enable_augmentation \
  --text_augmentation_type simple-paraphrase \
  --text_aggregation mean \
  --epochs 30
```

### Vision Ensemble Only
```bash
python train.py --dataset_name vivqa \
  --enable_augmentation \
  --image_augmentation_type multi-view \
  --vis_aggregation max \
  --epochs 30
```

### Full Multi-Input Ensemble
```bash
python train.py --dataset_name vivqa \
  --enable_augmentation \
  --text_augmentation_type simple-paraphrase \
  --image_augmentation_type multi-view \
  --text_aggregation mean \
  --vis_aggregation max \
  --enable_curriculum \
  --epochs 30
```

## Implementation Details

### Shape Compatibility

The model automatically detects input shapes:

**Text Encoder:**
- Input: `[batch, seq_len]` or `[batch, num_texts, seq_len]`
- Output: Always `[batch, hidden_size]`

**Vision Encoder:**
- Input: `[batch, C, H, W]` or `[batch, num_images, C, H, W]`
- Output: Always `[batch, hidden_size]`

### Backward Compatibility

✅ **Fully backward compatible** with existing code:
- Single-input augmentations work as before
- No configuration changes needed for existing experiments
- New features are opt-in via augmentation type

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**: Reduce effective batch size
```python
# If using 3 paraphrases and 3 views:
# Effective batch = batch_size × 3 × 3
# Reduce batch_size by factor of 9
batch_size = 32 // 9  # = 3 or 4
```

### Issue: Slow Training

**Solution 1**: Use offline paraphrases
```python
# Pre-generate paraphrases and cache
paraphrase_cache = load_paraphrases_from_file()
text_aug = ParaphraseTextAugmentation(paraphrase_cache=paraphrase_cache)
```

**Solution 2**: Reduce number of inputs
```python
num_paraphrases = 2  # Instead of 3-5
num_views = 2        # Instead of 3-4
```

### Issue: Tokenizer Batch Error

**Problem**: Some tokenizers don't handle list of lists well

**Solution**: Already handled in dataset - tokenizer receives flat list

## References

- Example script: `examples/multi_input_usage.py`
- Paraphrase augmentation: `augmentation/textual/paraphrase.py`
- Multi-view augmentation: `augmentation/visual/multi_view.py`
- Model implementation: `models/simple_vqa.py`
- Dataset implementation: `dataset/base.py`
