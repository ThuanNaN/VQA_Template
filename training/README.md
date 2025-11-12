# Training Module

This module provides an Object-Oriented Programming (OOP) architecture for training VQA models with support for:

- ✅ Curriculum Learning
- ✅ Dynamic Augmentation
- ✅ Type-safe Configuration
- ✅ Extensible Architecture
- ✅ Clean Separation of Concerns

## Quick Start

### Basic Training

```bash
python train.py --dataset_name vivqa --epochs 30
```

### With Augmentation

```bash
python train.py \
    --dataset_name vivqa \
    --enable_augmentation \
    --augmentation_type masked \
    --epochs 30
```

### With Curriculum Learning

```bash
python train.py \
    --dataset_name vivqa \
    --enable_augmentation \
    --enable_curriculum \
    --epochs 30 \
    --easy_epochs 9 \
    --medium_epochs 9 \
    --hard_epochs 12
```

## Components

### 1. Configuration (`config.py`)

Type-safe dataclasses for managing all configurations:

- `ModelConfig` - Model architecture settings
- `DataConfig` - Dataset and data loading
- `AugmentationConfig` - Augmentation and curriculum
- `TrainingConfig` - Training hyperparameters
- `ExperimentConfig` - Complete experiment setup

### 2. Trainer (`trainer.py`)

Custom trainer extending HuggingFace's Trainer:

- `VQATrainer` - Main trainer with curriculum support
- `CurriculumLearningCallback` - Manages difficulty progression

### 3. Pipeline (`pipeline.py`)

High-level orchestration:

- `VQATrainingPipeline` - Complete training workflow
- `DatasetFactory` - Dataset creation

## Architecture Benefits

1. **Extensibility** - Easy to add new components
2. **Maintainability** - Clear code organization
3. **Reusability** - Modular components
4. **Type Safety** - Better IDE support
5. **Scalability** - Ready for complex experiments

## Examples

See `examples/training_oop_examples.py` for detailed examples of:

- Basic augmentation
- Curriculum learning
- Programmatic configuration
- Custom augmentation strategies
- Dataset integration

## Documentation

See `docs/OOP_ARCHITECTURE.md` for complete documentation.
