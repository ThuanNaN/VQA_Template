# Wrong Prediction Tracking Feature

## Overview

The Wrong Prediction Tracking feature automatically monitors and saves all incorrectly predicted samples during validation/evaluation. This helps researchers and developers:

- **Identify problematic samples** where the model consistently fails
- **Monitor model improvement** across training epochs
- **Analyze error patterns** to guide model refinement
- **Debug dataset issues** by examining frequently misclassified samples

## Features

- **Automatic tracking**: Captures all wrong predictions during each validation epoch
- **Detailed metadata**: Saves questions, predictions, ground truth, and confidence scores
- **Top-5 predictions**: Includes top-5 predicted answers with scores for deeper analysis
- **Dual export formats**: JSON for programmatic access and human-readable text summaries
- **Zero configuration**: Enabled by default, works out of the box

## Usage

### Basic Usage (Default Enabled)

The feature is enabled by default. Simply run training as usual:

```bash
python train.py \
    --dataset_name vivqa \
    --epochs 10 \
    --batch_size 32
```

Wrong predictions will be saved to `wrong_predictions/{dataset}-{run_name}-{seed}/epoch_XXX/`

### Custom Configuration

To customize the wrong prediction directory:

```bash
python train.py \
    --dataset_name vivqa \
    --wrong_prediction_dir custom_wrong_preds \
    --epochs 10
```

To disable wrong prediction tracking:

```bash
python train.py \
    --dataset_name vivqa \
    --disable_wrong_prediction_tracking \
    --epochs 10
```

## Output Structure

```
wrong_predictions/
└── vivqa-run-71/
    ├── epoch_000/
    │   ├── wrong_predictions.json          # Detailed JSON export
    │   └── wrong_predictions_summary.txt   # Human-readable summary
    ├── epoch_001/
    │   ├── wrong_predictions.json
    │   └── wrong_predictions_summary.txt
    └── ...
```

### JSON Format

The JSON file contains structured data for each wrong prediction:

```json
{
  "epoch": 0,
  "num_wrong_predictions": 42,
  "wrong_samples": [
    {
      "sample_idx": 123,
      "image_path": "data/vivqa/images/COCO_train2014_000000123456.jpg",
      "question": "Màu sắc của chiếc xe là gì?",
      "model_prediction": "Xanh",
      "ground_truth": "Đỏ",
      "prediction_label": 2,
      "ground_truth_label": 3,
      "top5_predictions": [
        {"rank": 1, "answer": "Xanh", "label": 2, "score": 0.8234},
        {"rank": 2, "answer": "Đỏ", "label": 3, "score": 0.1234},
        {"rank": 3, "answer": "Vàng", "label": 4, "score": 0.0432},
        {"rank": 4, "answer": "Trắng", "label": 5, "score": 0.0089},
        {"rank": 5, "answer": "Đen", "label": 6, "score": 0.0011}
      ]
    },
    ...
  ]
}
```

### Text Summary Format

The text file provides an easy-to-read overview:

```
Epoch 0 - Wrong Predictions Summary
================================================================================

Total wrong predictions: 42

--------------------------------------------------------------------------------

[1] Sample Index: 123
Image: data/vivqa/images/COCO_train2014_000000123456.jpg
Question: Màu sắc của chiếc xe là gì?
Ground Truth: Đỏ
Prediction: Xanh
Top-5 Predictions:
  1. Xanh (score: 0.8234)
  2. Đỏ (score: 0.1234)
  3. Vàng (score: 0.0432)
  4. Trắng (score: 0.0089)
  5. Đen (score: 0.0011)
--------------------------------------------------------------------------------

[2] Sample Index: 456
...
```

## Analysis Examples

### Finding Consistent Errors

Track samples that are wrong across multiple epochs:

```python
import json
from pathlib import Path

def find_persistent_errors(wrong_pred_dir, min_epochs=3):
    """Find samples incorrectly predicted in multiple epochs."""
    epoch_dirs = sorted(Path(wrong_pred_dir).glob('epoch_*'))
    
    # Collect wrong sample indices per epoch
    wrong_by_epoch = {}
    for epoch_dir in epoch_dirs:
        json_file = epoch_dir / 'wrong_predictions.json'
        with open(json_file) as f:
            data = json.load(f)
        epoch = data['epoch']
        wrong_by_epoch[epoch] = {s['sample_idx'] for s in data['wrong_samples']}
    
    # Find samples wrong in multiple epochs
    all_wrong_indices = set.union(*wrong_by_epoch.values())
    persistent_errors = {}
    
    for idx in all_wrong_indices:
        count = sum(1 for wrong_set in wrong_by_epoch.values() if idx in wrong_set)
        if count >= min_epochs:
            persistent_errors[idx] = count
    
    return persistent_errors

# Usage
errors = find_persistent_errors('wrong_predictions/vivqa-run-71', min_epochs=3)
print(f"Found {len(errors)} samples with persistent errors")
```

### Tracking Error Rate Over Time

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt

def plot_error_rate(wrong_pred_dir, total_val_samples):
    """Plot error rate progression across epochs."""
    epoch_dirs = sorted(Path(wrong_pred_dir).glob('epoch_*'))
    
    epochs = []
    error_rates = []
    
    for epoch_dir in epoch_dirs:
        json_file = epoch_dir / 'wrong_predictions.json'
        with open(json_file) as f:
            data = json.load(f)
        
        epochs.append(data['epoch'])
        error_rate = data['num_wrong_predictions'] / total_val_samples
        error_rates.append(error_rate)
    
    plt.plot(epochs, error_rates, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.title('Validation Error Rate Over Time')
    plt.grid(True)
    plt.savefig('error_rate.png')

# Usage
plot_error_rate('wrong_predictions/vivqa-run-71', total_val_samples=1000)
```

## Integration with Training Pipeline

The feature integrates seamlessly with the existing training pipeline:

1. **VQATrainingPipeline** creates the `WrongPredictionTracker` if enabled
2. **VQATrainer** sets up the `WrongPredictionCallback`
3. **WrongPredictionCallback** triggers on each validation evaluation
4. Wrong predictions are automatically saved after each epoch

## Configuration Options

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Wrong prediction tracking
    enable_wrong_prediction_tracking: bool = True
    wrong_prediction_dir: str = 'wrong_predictions'
```

### CLI Arguments

- `--enable_wrong_prediction_tracking`: Enable tracking (default: True)
- `--disable_wrong_prediction_tracking`: Disable tracking
- `--wrong_prediction_dir DIR`: Custom output directory (default: 'wrong_predictions')

## Technical Details

### WrongPredictionTracker Class

Located in `utils/visualization.py`:

- **Purpose**: Manages storage and formatting of wrong predictions
- **Methods**:
  - `save_wrong_predictions()`: Save wrong predictions for an epoch
  - Automatic JSON and text export
  - Top-5 prediction analysis

### WrongPredictionCallback Class

Located in `training/trainer.py`:

- **Purpose**: HuggingFace Trainer callback to collect wrong predictions
- **Trigger**: `on_evaluate()` hook after each validation
- **Process**:
  1. Iterates through validation dataset
  2. Runs inference for each sample
  3. Compares predictions to ground truth
  4. Collects all mismatched samples
  5. Passes to tracker for storage

## Performance Considerations

- **Storage**: Minimal - only metadata and logits (not images)
- **Runtime**: Adds ~1-2% overhead during validation
- **Memory**: Processes samples one at a time to minimize memory usage

## Best Practices

1. **Review regularly**: Check wrong predictions after every few epochs
2. **Look for patterns**: Identify common error types (e.g., color confusion, counting errors)
3. **Validate dataset**: Some "wrong" predictions might indicate annotation errors
4. **Use for debugging**: If validation accuracy drops, examine which samples became wrong
5. **Compare across runs**: Track how different hyperparameters affect error patterns

## FAQ

**Q: Does this slow down training?**  
A: Minimal impact (~1-2% overhead during validation only).

**Q: How much disk space does it use?**  
A: Very little - only text metadata. For 1000 validation samples with 10% error rate, expect ~500KB per epoch.

**Q: Can I analyze wrong predictions programmatically?**  
A: Yes! The JSON format is designed for easy parsing and analysis.

**Q: What if I want to save images of wrong predictions?**  
A: The `image_path` is included in the JSON. You can write a script to copy/visualize them.

**Q: Does it work with all datasets?**  
A: Yes, it works with any dataset that has a label encoder (ViVQA, OpenViVQA, etc.).

## Related Features

- **Sample Observation**: Random sample tracking (use `--enable_sample_observation`)
- **Curriculum Learning**: Progressive difficulty adjustment
- **WandB Integration**: Combine with `--report_to_wandb` for cloud tracking
