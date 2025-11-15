# Aggregation Methods for VQA

This document describes the aggregation methods available for combining multiple text/image embeddings in the VQA framework.

## Overview

When using multiple paraphrases (text) or augmented views (images), we need to aggregate their embeddings into a single representation. The framework supports both simple and complex aggregation methods.

## Simple Aggregation Methods

These are parameter-free methods specified as strings:

### 1. Mean (`'mean'`)
- **Description**: Average all embeddings
- **Use case**: Balanced representation, reduces noise
- **Parameters**: None
```python
config = SimpleVQAConfig(
    text_aggregation='mean',
    vis_aggregation='mean'
)
```

### 2. Sum (`'sum'`)
- **Description**: Sum all embeddings
- **Use case**: Accumulate information from all inputs
- **Parameters**: None
```python
config = SimpleVQAConfig(text_aggregation='sum')
```

### 3. Max (`'max'`)
- **Description**: Element-wise maximum across embeddings
- **Use case**: Keep strongest signal for each dimension
- **Parameters**: None
```python
config = SimpleVQAConfig(text_aggregation='max')
```

### 4. First (`'first'`)
- **Description**: Take only the first embedding
- **Use case**: Baseline comparison, when order matters
- **Parameters**: None
```python
config = SimpleVQAConfig(text_aggregation='first')
```

## Learnable Aggregation Methods

These methods have trainable parameters and can learn optimal aggregation strategies.

### 5. Attention (`'attention'`)
- **Description**: Learnable attention weights over embeddings
- **Architecture**: Multi-head attention with learnable query
- **Parameters**: 
  - `num_heads`: Number of attention heads (default: 1)
- **Use case**: Learn which paraphrases/views are most important
```python
config = SimpleVQAConfig(
    text_aggregation='attention',
    text_aggregation_kwargs={'num_heads': 4}
)
```

### 6. Transformer (`'transformer'`)
- **Description**: Full transformer encoder with CLS token
- **Architecture**: Multi-layer transformer + CLS pooling
- **Parameters**:
  - `num_layers`: Transformer layers (default: 2)
  - `num_heads`: Attention heads (default: 4)
  - `dropout`: Dropout rate (default: 0.1)
- **Use case**: Complex interactions between paraphrases/views
```python
config = SimpleVQAConfig(
    text_aggregation='transformer',
    text_aggregation_kwargs={
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1
    }
)
```

### 7. Gated (`'gated'`)
- **Description**: Learnable gate network for weighted sum
- **Architecture**: MLP gate → softmax → weighted sum
- **Parameters**: None (hidden size determined by encoder)
- **Use case**: Adaptive importance weighting
```python
config = SimpleVQAConfig(text_aggregation='gated')
```

### 8. Weighted (`'weighted'`)
- **Description**: Learnable fixed weights per position
- **Architecture**: Position-specific learnable weights
- **Parameters**:
  - `max_num_items`: Maximum number of items to aggregate (default: 10)
- **Use case**: When paraphrase quality varies by position
```python
config = SimpleVQAConfig(
    text_aggregation='weighted',
    text_aggregation_kwargs={'max_num_items': 5}
)
```

## Custom Aggregators

You can create custom aggregators by inheriting from `BaseAggregator`:

```python
from models.aggregators import BaseAggregator
import torch.nn as nn

class MyCustomAggregator(BaseAggregator):
    def __init__(self, hidden_size: int):
        super().__init__()
        # Your custom layers here
        self.custom_layer = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: [batch_size, num_items, hidden_size]
        # Your custom aggregation logic
        return aggregated  # [batch_size, hidden_size]

# Use it
config = SimpleVQAConfig(
    text_aggregation=MyCustomAggregator(hidden_size=512)
)
```

## Usage Examples

### Example 1: Simple Mean Aggregation
```python
from models import SimpleVQA, SimpleVQAConfig

config = SimpleVQAConfig(
    vis_model_name='google/vit-base-patch16-224',
    text_model_name='vinai/bartpho-syllable-base',
    num_classes=1000,
    hidden_size=512,
    text_aggregation='mean',
    vis_aggregation='mean'
)

model = SimpleVQA(config)
```

### Example 2: Attention for Text, Mean for Vision
```python
config = SimpleVQAConfig(
    vis_model_name='google/vit-base-patch16-224',
    text_model_name='vinai/bartpho-syllable-base',
    num_classes=1000,
    hidden_size=512,
    text_aggregation='attention',
    vis_aggregation='mean',
    text_aggregation_kwargs={'num_heads': 8}
)

model = SimpleVQA(config)
```

### Example 3: Complex Aggregation for Both Modalities
```python
config = SimpleVQAConfig(
    vis_model_name='google/vit-base-patch16-224',
    text_model_name='vinai/bartpho-syllable-base',
    num_classes=1000,
    hidden_size=512,
    text_aggregation='transformer',
    vis_aggregation='gated',
    text_aggregation_kwargs={
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1
    }
)

model = SimpleVQA(config)
```

## Performance Considerations

| Method | Parameters | Speed | Memory | Best For |
|--------|-----------|-------|--------|----------|
| mean/sum/max/first | 0 | ⚡⚡⚡ | 💾 | Baseline, fast inference |
| attention | ~H²/k | ⚡⚡ | 💾💾 | Balanced performance |
| weighted | N | ⚡⚡⚡ | 💾 | Fixed position importance |
| gated | ~H²/2 | ⚡⚡ | 💾💾 | Adaptive weighting |
| transformer | ~4H²L | ⚡ | 💾💾💾 | Complex interactions |

*H = hidden_size, N = max_num_items, L = num_layers, k = num_heads*

## Training Tips

1. **Start Simple**: Begin with `'mean'` aggregation to establish baseline
2. **Add Complexity**: Try `'attention'` or `'gated'` for learnable aggregation
3. **Monitor Overfitting**: Complex aggregators may overfit on small datasets
4. **Mix Strategies**: Use complex aggregation for text, simple for vision (or vice versa)
5. **Hyperparameter Tuning**: Number of heads/layers affects performance significantly

## When to Use Which Method

- **Single input (no aggregation needed)**: All methods work, use `'first'`
- **2-3 paraphrases**: `'mean'` or `'attention'`
- **5+ paraphrases**: `'transformer'` or `'gated'`
- **Ordered paraphrases** (by quality): `'weighted'`
- **Limited compute**: `'mean'`, `'sum'`, or `'max'`
- **Research/experimentation**: `'transformer'` or custom aggregator
