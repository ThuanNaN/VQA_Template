# ViVQA Augmentation Experiments

This directory contains comprehensive experiments to evaluate the impact of different augmentation strategies on VQA performance using the ViVQA dataset.

## Experiments Overview

The experiment suite includes **7 experiments** that systematically test:

1. **Baseline** - No augmentation (control group)
2. **Text Augmentation** - Text augmentation only
3. **Image Augmentation** - Image augmentation only
4. **Text + Image** - Combined text and image augmentation
5. **Text + CL** - Text augmentation with Curriculum Learning
6. **Image + CL** - Image augmentation with Curriculum Learning
7. **Full (Text + Image + CL)** - All augmentation strategies combined

## Quick Start

### Run All Experiments

```bash
./experiments/run_vivqa_augmentation_experiments.sh
```

This will run all 7 experiments sequentially. Each experiment trains for 30 epochs.

### Configuration

Edit the script to modify:

```bash
# Dataset and model
DATASET_NAME="vivqa"
VIS_MODEL="google/vit-base-patch16-224"
TEXT_MODEL="vinai/bartpho-syllable-base"

# Training parameters
EPOCHS=30
BATCH_SIZE=64
LEARNING_RATE=1e-4

# Curriculum Learning (9 easy + 9 medium + 12 hard = 30 total)
EASY_EPOCHS=9
MEDIUM_EPOCHS=9
HARD_EPOCHS=12

# WandB logging
ENABLE_WANDB=true
WANDB_PROJECT="VQA-Augmentation-Experiments"
```

## Experiment Details

### Experiment 1: Baseline
```bash
python train.py \
    --dataset_name vivqa \
    --epochs 30 \
    --run_name exp1_baseline
```
- **Purpose**: Establish baseline performance
- **Augmentation**: None
- **Expected**: Lower performance but faster convergence initially

### Experiment 2: Text Augmentation
```bash
python train.py \
    --dataset_name vivqa \
    --enable_text_augmentation \
    --text_augmentation_type simple \
    --epochs 30 \
    --run_name exp2_text_augment
```
- **Purpose**: Test impact of text augmentation alone
- **Augmentation**: Simple text augmentation (word deletion, swapping, synonym replacement)
- **Expected**: Better generalization on questions

### Experiment 3: Image Augmentation
```bash
python train.py \
    --dataset_name vivqa \
    --enable_image_augmentation \
    --image_augmentation_type masked \
    --patch_size 16 \
    --epochs 30 \
    --run_name exp3_image_augment
```
- **Purpose**: Test impact of image augmentation alone
- **Augmentation**: Masked image augmentation (MAE-inspired)
- **Expected**: Better generalization on images, robust to occlusion

### Experiment 4: Text + Image Augmentation
```bash
python train.py \
    --dataset_name vivqa \
    --enable_text_augmentation \
    --enable_image_augmentation \
    --epochs 30 \
    --run_name exp4_text_image_augment
```
- **Purpose**: Test combined effect of both augmentations
- **Augmentation**: Both text and image
- **Expected**: Best generalization without curriculum learning

### Experiment 5: Text Augmentation + Curriculum Learning
```bash
python train.py \
    --dataset_name vivqa \
    --enable_text_augmentation \
    --enable_curriculum \
    --easy_epochs 9 \
    --medium_epochs 9 \
    --hard_epochs 12 \
    --epochs 30 \
    --run_name exp5_text_augment_cl
```
- **Purpose**: Test curriculum learning with text augmentation
- **Augmentation**: Text with progressive difficulty
- **Schedule**: 
  - Epochs 0-8: EASY (20% apply probability)
  - Epochs 9-17: MEDIUM (50% apply probability)
  - Epochs 18-29: HARD (80% apply probability)
- **Expected**: Better training stability and convergence

### Experiment 6: Image Augmentation + Curriculum Learning
```bash
python train.py \
    --dataset_name vivqa \
    --enable_image_augmentation \
    --enable_curriculum \
    --easy_epochs 9 \
    --medium_epochs 9 \
    --hard_epochs 12 \
    --epochs 30 \
    --run_name exp6_image_augment_cl
```
- **Purpose**: Test curriculum learning with image augmentation
- **Augmentation**: Masked images with progressive difficulty
- **Schedule**:
  - Epochs 0-8: EASY (15% mask ratio)
  - Epochs 9-17: MEDIUM (50% mask ratio)
  - Epochs 18-29: HARD (75% mask ratio)
- **Expected**: Gradual learning from easy to hard visual features

### Experiment 7: Full Augmentation + Curriculum Learning ⭐
```bash
python train.py \
    --dataset_name vivqa \
    --enable_text_augmentation \
    --enable_image_augmentation \
    --enable_curriculum \
    --easy_epochs 9 \
    --medium_epochs 9 \
    --hard_epochs 12 \
    --epochs 30 \
    --run_name exp7_full_augment_cl
```
- **Purpose**: Test the complete augmentation strategy
- **Augmentation**: Both text and image with curriculum learning
- **Expected**: Best overall performance with improved generalization

## Output Structure

```
runs/vivqa_augmentation_experiments/
├── exp1_baseline/
│   ├── checkpoint-*/
│   ├── logs/
│   └── config.json
├── exp2_text_augment/
├── exp3_image_augment/
├── exp4_text_image_augment/
├── exp5_text_augment_cl/
├── exp6_image_augment_cl/
└── exp7_full_augment_cl/
```

## Monitoring

### With WandB (Recommended)

If `ENABLE_WANDB=true`, view real-time results at:
https://wandb.ai (Project: VQA-Augmentation-Experiments)

Compare all experiments side-by-side:
- Training loss curves
- Validation accuracy
- Curriculum difficulty progression
- Augmentation parameters

### Local Logs

Check training logs:
```bash
tail -f runs/vivqa_augmentation_experiments/exp*/logs/train.log
```

## Expected Results

Based on curriculum learning research, we expect:

| Experiment | Training Speed | Generalization | Final Accuracy | Robustness |
|------------|---------------|----------------|----------------|------------|
| 1. Baseline | ⭐⭐⭐ | ⭐⭐ | Baseline | ⭐⭐ |
| 2. Text Aug | ⭐⭐ | ⭐⭐⭐ | +2-3% | ⭐⭐⭐ |
| 3. Image Aug | ⭐⭐ | ⭐⭐⭐ | +3-5% | ⭐⭐⭐⭐ |
| 4. Text+Image | ⭐⭐ | ⭐⭐⭐⭐ | +4-6% | ⭐⭐⭐⭐ |
| 5. Text+CL | ⭐⭐⭐ | ⭐⭐⭐⭐ | +3-4% | ⭐⭐⭐⭐ |
| 6. Image+CL | ⭐⭐⭐ | ⭐⭐⭐⭐ | +5-7% | ⭐⭐⭐⭐⭐ |
| 7. Full+CL ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | +6-9% | ⭐⭐⭐⭐⭐ |

## Analysis

### Compare Results

After experiments complete, use this script to compare:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results from WandB or local logs
experiments = [
    'exp1_baseline',
    'exp2_text_augment',
    'exp3_image_augment',
    'exp4_text_image_augment',
    'exp5_text_augment_cl',
    'exp6_image_augment_cl',
    'exp7_full_augment_cl',
]

# Compare final accuracies
# Plot training curves
# Analyze convergence speed
```

### Key Metrics to Compare

1. **Final Validation Accuracy** - Overall performance
2. **Training Stability** - Loss variance over epochs
3. **Convergence Speed** - Epochs to reach peak performance
4. **Generalization Gap** - Train vs validation accuracy difference
5. **Robustness** - Performance on challenging samples

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
BATCH_SIZE=32  # or 16
```

### Slow Training
Enable mixed precision:
```bash
FP16_FLAG="--fp16"
```

Increase workers:
```bash
DATALOADER_WORKERS=8
```

### Resume Failed Experiment
```bash
python train.py \
    --dataset_name vivqa \
    --run_name exp3_image_augment \
    --resume_from_checkpoint runs/vivqa_augmentation_experiments/exp3_image_augment/checkpoint-*
```

## Time Estimates

Per experiment (30 epochs):
- **GPU (V100/A100)**: ~2-3 hours
- **GPU (RTX 3090)**: ~3-4 hours
- **GPU (RTX 2080 Ti)**: ~4-6 hours

Total time for all 7 experiments:
- **~14-42 hours** (depending on GPU)

Run overnight or on weekends for best efficiency.

## Next Steps

After experiments complete:

1. **Analyze Results**: Compare metrics in WandB
2. **Statistical Significance**: Run t-tests on final accuracies
3. **Ablation Studies**: Test different curriculum schedules
4. **Hyperparameter Tuning**: Optimize learning rate, augmentation strength
5. **Test on Other Datasets**: Try OpenViVQA, ViTextVQA, etc.

## Citation

If you use this experimental setup, please cite:

```bibtex
@misc{vqa_augmentation_experiments,
  title={Augmentation and Curriculum Learning for Vietnamese Visual Question Answering},
  author={Your Name},
  year={2025}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
