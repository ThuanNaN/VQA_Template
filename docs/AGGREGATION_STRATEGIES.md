# Aggregation Strategies for Multiple Inputs

This document explains how to use different aggregation strategies when working with multiple text inputs (paraphrases) or multiple image inputs (augmentations).

## Overview

When you have multiple paraphrases of a question or multiple augmented versions of an image, you need to combine their embeddings. This framework supports both **simple** and **complex** aggregation methods.

## Quick Start

### Command Line
```bash
# Simple mean aggregation (default)
python train.py --dataset_name vivqa --text_aggregation mean --vis_aggregation mean

# Attention-based aggregation (learnable)
python train.py --dataset_name vivqa --text_aggregation attention --vis_aggregation attention

# Mixed: Transformer for text, mean for images
python train.py --dataset_name vivqa --text_aggregation transformer --vis_aggregation mean
```

### Programmatic Configuration
```python
from training import ExperimentConfig, ModelConfig

config = ExperimentConfig(
    model=ModelConfig(
        text_aggregation='attention',  # Learnable attention weights
        vis_aggregation='mean',        # Simple averaging
        text_aggregation_kwargs={
            'num_heads': 4,
            'dropout': 0.1
        }
    )
)
```

## Available Aggregation Methods

### Simple Aggregation (Non-learnable)

#### 1. Mean (`'mean'`)
- **Description**: Average all embeddings
- **Parameters**: None
- **Complexity**: O(1)
- **Best for**: Quick baseline, stable training
```python
text_aggregation='mean'
```

#### 2. Sum (`'sum'`)
- **Description**: Sum all embeddings
- **Parameters**: None
- **Complexity**: O(1)
- **Best for**: When scale matters (e.g., counting)
```python
text_aggregation='sum'
```

#### 3. Max (`'max'`)
- **Description**: Element-wise maximum
- **Parameters**: None
- **Complexity**: O(1)
- **Best for**: Sparse features, highlighting strongest signals
```python
text_aggregation='max'
```

#### 4. First (`'first'`)
- **Description**: Use only the first input
- **Parameters**: None
- **Complexity**: O(1)
- **Best for**: Fallback when no aggregation needed
```python
text_aggregation='first'
```

---

### Complex Aggregation (Learnable)

#### 1. Weighted (`'weighted'`)
- **Description**: Learnable scalar weights per input
- **Parameters**: 
  - `dropout` (default: 0.1)
- **Complexity**: O(n) - Low
- **Best for**: Simple learnable combination
```python
text_aggregation='weighted'
text_aggregation_kwargs={'dropout': 0.1}
```

#### 2. Attention (`'attention'`)
- **Description**: Multi-head self-attention aggregation
- **Parameters**:
  - `num_heads` (default: 4)
  - `dropout` (default: 0.1)
- **Complexity**: O(n) - Medium
- **Best for**: Learning to weight inputs based on content
```python
text_aggregation='attention'
text_aggregation_kwargs={
    'num_heads': 4,
    'dropout': 0.1
}
```

#### 3. Gated (`'gated'`)
- **Description**: Gated fusion with tanh activation
- **Parameters**:
  - `dropout` (default: 0.1)
- **Complexity**: O(n) - Medium
- **Best for**: Filtering + combining with robustness
```python
text_aggregation='gated'
text_aggregation_kwargs={'dropout': 0.2}
```

#### 4. Transformer (`'transformer'`)
- **Description**: Full transformer encoder
- **Parameters**:
  - `num_layers` (default: 2)
  - `num_heads` (default: 4)
  - `dim_feedforward` (default: 2048)
  - `dropout` (default: 0.1)
- **Complexity**: O(n²) - High
- **Best for**: Complex reasoning over many paraphrases (3+)
```python
text_aggregation='transformer'
text_aggregation_kwargs={
    'num_layers': 2,
    'num_heads': 8,
    'dim_feedforward': 2048,
    'dropout': 0.1
}
```

## Use Cases

### Scenario 1: Paraphrase Ensembling (Text)
You have 3 paraphrases per question and want to learn which to emphasize:
```python
model=ModelConfig(
    text_aggregation='attention',
    text_aggregation_kwargs={'num_heads': 4}
)
```

### Scenario 2: Augmented Images (Vision)
You have 2-3 augmented versions of each image:
```python
model=ModelConfig(
    vis_aggregation='mean'  # Simple average works well
)
```

### Scenario 3: Many Paraphrases (5+)
You have many paraphrases and want deep reasoning:
```python
model=ModelConfig(
    text_aggregation='transformer',
    text_aggregation_kwargs={
        'num_layers': 2,
        'num_heads': 8
    }
)
```

### Scenario 4: Mixed Strategy
Complex aggregation for text, simple for vision:
```python
model=ModelConfig(
    text_aggregation='attention',
    vis_aggregation='mean',
    text_aggregation_kwargs={'num_heads': 4}
)
```

## Comparison Table

| Method      | Learnable | Parameters | GPU Memory | Best Use Case                    |
|-------------|-----------|------------|------------|----------------------------------|
| mean        | ❌        | 0          | Low        | Baseline, stable                 |
| sum         | ❌        | 0          | Low        | Scale-sensitive tasks            |
| max         | ❌        | 0          | Low        | Sparse features                  |
| first       | ❌        | 0          | Low        | Single input fallback            |
| weighted    | ✅        | Very Low   | Low        | Simple learnable weights         |
| attention   | ✅        | Low        | Medium     | Content-based weighting          |
| gated       | ✅        | Medium     | Medium     | Robust filtering + combination   |
| transformer | ✅        | High       | High       | Complex reasoning, many inputs   |

## Recommendations

1. **Start Simple**: Begin with `'mean'` aggregation as baseline
2. **Try Attention**: If you have 2-3 paraphrases, try `'attention'` for learnable weighting
3. **Go Complex**: For 5+ paraphrases, consider `'transformer'`
4. **Mix Strategies**: Use complex for text, simple for vision if resources are limited
5. **Monitor GPU**: Transformer aggregation increases memory usage

## Implementation Details

### How Aggregation Works

1. **Single Input**: If input is single (no paraphrases/augmentations), it passes through normally
   ```python
   # Input: [batch_size, seq_len]
   # Output: [batch_size, hidden_size]
   ```

2. **Multiple Inputs**: If input has multiple variants, aggregation is applied
   ```python
   # Input: [batch_size, num_variants, seq_len]
   # After encoding: [batch_size, num_variants, hidden_size]
   # After aggregation: [batch_size, hidden_size]
   ```

### Where Aggregation Happens

```python
# In models/simple_vqa.py
class TextEncoder(BaseTextEncoder):
    def forward(self, inputs):
        # 1. Encode all variants
        embeddings = self.encoder(...)  # [batch, num_variants, hidden]
        
        # 2. Aggregate using selected strategy
        aggregated = self.aggregator(embeddings)  # [batch, hidden]
        
        return aggregated
```

## Examples

See `examples/aggregation_usage.py` for complete working examples of all aggregation strategies.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multiple Text Inputs                         │
│  ["Xe màu gì?", "Màu sắc xe là gì?", "Chiếc xe có màu gì?"]  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Text Encoder    │
                    │  (BARTPho/BERT)  │
                    └──────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ Embeddings [batch, 3, hidden] │
              └───────────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │      Aggregation Strategy           │
            │  ┌───────┬──────────┬────────────┐  │
            │  │ Mean  │Attention │Transformer │  │
            │  └───────┴──────────┴────────────┘  │
            └─────────────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │ Aggregated [batch, hidden]   │
              └──────────────────────────────┘
                              │
                              ▼
                        Classifier
```

## Troubleshooting

### Out of Memory Error
- Reduce batch size
- Use simpler aggregation (`'mean'` instead of `'transformer'`)
- Reduce number of paraphrases/augmentations

### Slow Training
- Use `'mean'` or `'attention'` instead of `'transformer'`
- Reduce `num_layers` in transformer aggregation
- Use fewer paraphrases per sample

### Poor Performance
- Try learnable aggregation (`'attention'`, `'weighted'`)
- Increase model capacity (more heads, layers)
- Ensure paraphrases are diverse and meaningful

## Related Documentation

- [Model Architecture](ARCHITECTURE_DIAGRAM.md)
- [Multi-Input Dataset Guide](../augmentation/README.md)
- [Training Configuration](../training/README.md)
