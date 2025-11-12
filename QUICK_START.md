# Quick Start Guide - Training Architecture

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Simple Training (No Augmentation)

```bash
python train.py \
    --dataset_name vivqa \
    --epochs 30 \
    --batch_size 64
```

### 2. Training with Augmentation

```bash
python train.py \
    --dataset_name vivqa \
    --enable_augmentation \
    --augmentation_type masked \
    --patch_size 16 \
    --epochs 30
```

### 3. Training with Curriculum Learning ⭐ (Recommended)

```bash
python train.py \
    --dataset_name openvivqa \
    --enable_augmentation \
    --enable_curriculum \
    --epochs 50 \
    --easy_epochs 15 \
    --medium_epochs 15 \
    --hard_epochs 20 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --report_to_wandb \
    --wandb_name VQA-Experiments
```

## Key Features

✅ **Curriculum Learning** - Progressive difficulty (EASY → MEDIUM → HARD)
✅ **Dynamic Augmentation** - Automatically adjusts based on training epoch
✅ **Type-Safe Configuration** - Dataclasses for all configurations
✅ **Extensible Architecture** - Easy to add new augmentation strategies
✅ **WandB Integration** - Track experiments with Weights & Biases

## Programmatic Usage

```python
from training import (
    ExperimentConfig,
    ModelConfig,
    DataConfig,
    AugmentationConfig,
    TrainingConfig,
    VQATrainingPipeline,
)

# Create configuration
config = ExperimentConfig(
    model=ModelConfig(
        vis_model_name='google/vit-base-patch16-224',
        text_model_name='vinai/bartpho-syllable-base'
    ),
    data=DataConfig(
        dataset_name='vivqa',
        batch_size=32,
    ),
    augmentation=AugmentationConfig(
        enable_augmentation=True,
        enable_curriculum=True,
        augmentation_type='masked',
    ),
    training=TrainingConfig(
        epochs=30,
        learning_rate=5e-5,
        report_to_wandb=True,
    )
)

# Run training
pipeline = VQATrainingPipeline(config)
pipeline.run()
```

## Custom Augmentation

```python
from augmentation import BaseImageAugmentation, AugmentationFactory, DifficultyLevel
from PIL import Image, ImageFilter

class MyCustomAugmentation(BaseImageAugmentation):
    def _configure_parameters(self):
        # Configure based on difficulty
        if self.difficulty == DifficultyLevel.EASY:
            self.strength = 0.1
        elif self.difficulty == DifficultyLevel.MEDIUM:
            self.strength = 0.5
        else:  # HARD
            self.strength = 1.0
    
    def augment(self, image: Image.Image, **kwargs) -> Image.Image:
        # Your augmentation logic here
        return image
    
    def get_augmentation_info(self) -> dict:
        return {
            'type': 'MyCustom',
            'difficulty': self.difficulty.value,
            'strength': self.strength
        }

# Register and use
AugmentationFactory.register_image_augmentation('my_custom', MyCustomAugmentation)

augmentor = AugmentationFactory.create_image_augmentation(
    augmentation_type='my_custom',
    difficulty=DifficultyLevel.MEDIUM
)
```

## Available Command-Line Arguments

### Model Arguments

- `--vis_model_name` - Vision model (default: google/vit-base-patch16-224)
- `--text_model_name` - Text model (default: vinai/bartpho-syllable-base)

### Dataset Arguments

- `--dataset_name` - Dataset to use (vivqa, openvivqa, vitextvqa, etc.)
- `--batch_size` - Batch size (default: 64)
- `--seq_len` - Sequence length (default: 64)

### Training Arguments

- `--epochs` - Number of epochs (default: 30)
- `--learning_rate` - Learning rate (default: 1e-4)
- `--weight_decay` - Weight decay (default: 1e-4)
- `--fp16` - Enable mixed precision training

### Augmentation Arguments

- `--enable_augmentation` - Enable image augmentation
- `--augmentation_type` - Type of augmentation (masked, none)
- `--patch_size` - Patch size for masked augmentation (default: 16)

### Curriculum Learning Arguments

- `--enable_curriculum` - Enable curriculum learning
- `--easy_epochs` - Number of easy epochs
- `--medium_epochs` - Number of medium epochs
- `--hard_epochs` - Number of hard epochs

### Logging Arguments

- `--report_to_wandb` - Enable WandB logging
- `--wandb_name` - WandB project name
- `--run_name` - Run name for identification
- `--output_dir` - Output directory (default: runs)

## Directory Structure

```plaintext
VQA_Template/
├── train.py                    # Old training script (still works)
├── train.py                 # New OOP-based training script ⭐
├── augmentation/               # Augmentation module
│   ├── base.py                # Base classes
│   ├── factory.py             # Factory pattern
│   └── visual/mask.py         # Masked augmentation
├── training/                   # Training module ⭐
│   ├── config.py              # Configuration dataclasses
│   ├── trainer.py             # Custom VQA trainer
│   ├── pipeline.py            # Training pipeline
│   └── README.md
├── dataset/                    # Dataset module
│   └── base.py                # Base dataset (with augmentation support)
├── docs/
│   ├── OOP_ARCHITECTURE.md    # Complete documentation
│   └── ARCHITECTURE_DIAGRAM.md # Visual diagrams
└── examples/
    ├── training_oop_examples.py    # Usage examples
    └── training_commands.sh        # Example commands
```

## Documentation

- **Complete Guide**: `docs/OOP_ARCHITECTURE.md`
- **Architecture Diagrams**: `docs/ARCHITECTURE_DIAGRAM.md`
- **Training Module**: `training/README.md`
- **Refactoring Summary**: `REFACTORING_SUMMARY.md`

## Examples

See `examples/` directory for:

- `training_oop_examples.py` - Programmatic usage examples
- `training_commands.sh` - Command-line examples

## Testing

```bash
# Run augmentation tests
python tests/test_augmentation_oop.py
```

## Tips

1. **Start Small**: Test with a few epochs first

   ```bash
   python train.py --dataset_name vivqa --epochs 5
   ```

2. **Use Curriculum Learning**: It helps model convergence

   ```bash
   --enable_augmentation --enable_curriculum
   ```

3. **Monitor with WandB**: Track experiments

   ```bash
   --report_to_wandb --wandb_name MyProject
   ```

4. **Adjust Batch Size**: Based on GPU memory

   ```bash
   --batch_size 16  # For smaller GPUs
   ```

5. **Use Mixed Precision**: Faster training

   ```bash
   --fp16
   ```

## Common Issues

**Q: ModuleNotFoundError: No module named 'transformers'**
A: Install dependencies: `pip install -r requirements.txt`

**Q: CUDA out of memory**
A: Reduce batch size: `--batch_size 16` or `--batch_size 8`

**Q: Training is slow**
A: Use `--fp16` and `--dataloader_workers 4`

**Q: Want to try different augmentation strengths**
A: Use curriculum learning: `--enable_curriculum`

## Next Steps

1. **Try basic training**: `python train.py --dataset_name vivqa --epochs 5`
2. **Enable augmentation**: Add `--enable_augmentation`
3. **Add curriculum learning**: Add `--enable_curriculum`
4. **Track with WandB**: Add `--report_to_wandb`
5. **Customize**: Create your own augmentation strategies

## Support

For detailed documentation, see:

- `docs/OOP_ARCHITECTURE.md` - Complete architecture guide
- `docs/ARCHITECTURE_DIAGRAM.md` - Visual diagrams
- `examples/training_oop_examples.py` - Code examples

Happy training! 🚀
