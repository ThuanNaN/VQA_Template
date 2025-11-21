# VQA Augmentation Experiments

This directory contains comprehensive experiments to evaluate the impact of different augmentation strategies on VQA performance using Vietnamese VQA datasets (ViVQA and OpenViVQA).

## Experiments Overview

The experiment suite includes **7 experiments** that systematically test:

1. **Baseline** - No augmentation (control group)
2. **Text Augmentation** - Rule-based POS tagging text augmentation only
3. **Image Augmentation** - Masked patch image augmentation only
4. **Text + Image** - Combined text and image augmentation
5. **Text + CL** - Text augmentation with Curriculum Learning
6. **Image + CL** - Image augmentation with Curriculum Learning
7. **Full (Text + Image + CL)** - All augmentation strategies combined

## Quick Start

### Run All Experiments

For ViVQA dataset:
```bash
./experiments/run_vivqa.sh
```

For OpenViVQA dataset:
```bash
./experiments/run_openvivqa.sh
```

Each script runs all 7 experiments sequentially. Each experiment trains for 30 epochs.

### Configuration

Edit the script to modify:

```bash
# Dataset and model
DATASET_NAME="vivqa"  # or "openvivqa"
VIS_MODEL="google/vit-base-patch16-224"
TEXT_MODEL="vinai/bartpho-syllable-base"

# Training parameters
EPOCHS=30
BATCH_SIZE=64
LEARNING_RATE=1e-4

# Curriculum Learning Scheduler
CURRICULUM_STRATEGY="linear"  # Options: linear, cosine, exponential, step, polynomial
WARMUP_EPOCHS=0
CURRICULUM_GAMMA=0.1  # For exponential strategy
CURRICULUM_STEP_SIZE=10  # For step strategy
CURRICULUM_POWER=2.0  # For polynomial strategy

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
    --text_augmentation_type rule-based \
    --epochs 30 \
    --run_name exp2_text_augment
```
- **Purpose**: Test impact of text augmentation alone
- **Augmentation**: Rule-based POS tagging paraphrase (3 rules: synonym replacement, ADV movement, active-to-passive)
- **Library**: underthesea for Vietnamese NLP (POS tagging, word tokenization)
- **Expected**: Better generalization on questions with diverse phrasings

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
    --curriculum_strategy linear \
    --epochs 30 \
    --run_name exp5_text_augment_cl
```
- **Purpose**: Test curriculum learning with text augmentation
- **Augmentation**: Text with 3-phase progressive difficulty
- **Phases** (based on difficulty 0.0-1.0):
  - Phase 1 (difficulty < 0.33): EASY - Generate 2 paraphrases (3 total with original)
  - Phase 2 (0.33 ≤ difficulty < 0.66): MEDIUM - Generate 1 paraphrase (2 total with original)
  - Phase 3 (difficulty ≥ 0.66): HARD - No augmentation (1 total - original only)
- **Scheduler**: CurriculumScheduler with linear strategy calculates difficulty automatically
- **Expected**: Better training stability and convergence with smooth difficulty progression

### Experiment 6: Image Augmentation + Curriculum Learning
```bash
python train.py \
    --dataset_name vivqa \
    --enable_image_augmentation \
    --enable_curriculum \
    --curriculum_strategy linear \
    --epochs 30 \
    --run_name exp6_image_augment_cl
```
- **Purpose**: Test curriculum learning with image augmentation
- **Augmentation**: Masked patch images with 3-phase progressive difficulty
- **Phases** (based on difficulty 0.0-1.0):
  - Phase 1 (difficulty < 0.33): EASY - Mask 15% of patches
  - Phase 2 (0.33 ≤ difficulty < 0.66): MEDIUM - Mask 50% of patches
  - Phase 3 (difficulty ≥ 0.66): HARD - Mask 75% of patches
- **Scheduler**: CurriculumScheduler with linear strategy calculates difficulty automatically
- **Expected**: Gradual learning from easy to hard visual features with smooth progression

### Experiment 7: Full Augmentation + Curriculum Learning ⭐
```bash
python train.py \
    --dataset_name vivqa \
    --enable_text_augmentation \
    --enable_image_augmentation \
    --enable_curriculum \
    --curriculum_strategy linear \
    --epochs 30 \
    --run_name exp7_full_augment_cl
```
- **Purpose**: Test the complete augmentation strategy
- **Augmentation**: Both text and image with synchronized 3-phase curriculum learning
- **Text Phases**:
  - EASY (difficulty < 0.33): 2 paraphrases
  - MEDIUM (0.33 ≤ difficulty < 0.66): 1 paraphrase
  - HARD (difficulty ≥ 0.66): 0 paraphrases
- **Image Phases**:
  - EASY (difficulty < 0.33): 15% mask ratio
  - MEDIUM (0.33 ≤ difficulty < 0.66): 50% mask ratio
  - HARD (difficulty ≥ 0.66): 75% mask ratio
- **Expected**: Best overall performance with improved generalization and training stability

## Output Structure

```
observations/
├── vivqa-exp1_baseline-42/
│   ├── checkpoint-*/
│   ├── logs/
│   └── config.json
├── vivqa-exp2_text_augment-42/
├── vivqa-exp3_image_augment-42/
├── vivqa-exp4_text_image_augment-42/
├── vivqa-exp5_text_augment_cl-42/
├── vivqa-exp6_image_augment_cl-42/
├── vivqa-exp7_full_augment_cl-42/
├── openvivqa-exp1_baseline-42/
├── openvivqa-exp2_text_augment-42/
└── ...
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
tail -f observations/vivqa-exp*/logs/train.log
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
    --resume_from_checkpoint observations/vivqa-exp3_image_augment-42/checkpoint-*
```

## Time Estimates

Per experiment (30 epochs):
- **GPU (V100/A100)**: ~2-3 hours
- **GPU (RTX 3090)**: ~3-4 hours
- **GPU (RTX 2080 Ti)**: ~4-6 hours

Total time for all 7 experiments:
- **~14-42 hours** (depending on GPU)

Run overnight or on weekends for best efficiency.

## Curriculum Learning Details

### Scheduler Strategies

The experiments support 5 curriculum learning strategies:

1. **Linear** (Default): Difficulty increases linearly from 0 to 1
2. **Cosine**: Smooth cosine-based progression
3. **Exponential**: Rapid initial increase, then slows down
4. **Step**: Discrete steps at specified intervals
5. **Polynomial**: Polynomial-based progression (configurable power)

### Difficulty Calculation

The `CurriculumScheduler` automatically calculates difficulty for each epoch:

```python
from augmentation.scheduler import CurriculumScheduler

scheduler = CurriculumScheduler(
    total_epochs=30,
    strategy='linear',
    warmup_epochs=0
)

difficulty = scheduler.get_difficulty(current_epoch)
# Returns value between 0.0 and 1.0
```

### Analysis Tools

Use the provided analysis script to visualize curriculum learning:

```bash
python check_cl_relationship.py
```

This generates:
- Difficulty progression charts for all 5 strategies
- Text augmentation count vs difficulty
- Image mask ratio vs difficulty
- Detailed epoch-by-epoch analysis

## Next Steps

After experiments complete:

1. **Analyze Results**: Compare metrics in WandB
2. **Statistical Significance**: Run t-tests on final accuracies
3. **Ablation Studies**: Test different curriculum strategies (cosine, exponential, etc.)
4. **Hyperparameter Tuning**: Optimize learning rate, warmup epochs, curriculum power
5. **Test on Other Datasets**: Try ViTextVQA, ViOCRVQA, EVJVQA

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
