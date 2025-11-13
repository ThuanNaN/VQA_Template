# VQA Template - AI Coding Agent Instructions

## Project Overview
Vietnamese Visual Question Answering (VQA) research framework with curriculum learning, dynamic augmentation, and multi-dataset support (ViVQA, OpenViVQA, ViTextVQA, ViOCRVQA, EVJVQA, ViVQAX).

## Architecture Pattern: OOP with Factory Pattern

### Core Design Principles
1. **Dataclass Configuration**: All configs use `@dataclass` (see `training/config.py`) - never use dictionaries for configuration
2. **Factory Pattern**: Use `AugmentationFactory` and `DatasetFactory` to create instances - never instantiate augmentation/dataset classes directly
3. **Smooth Difficulty Values**: Augmentation difficulty is **continuous float 0.0-1.0**, not discrete "EASY/MEDIUM/HARD"
4. **Base Classes**: All custom augmentations must inherit from `BaseImageAugmentation` or `BaseTextAugmentation` and implement `_configure_parameters()` and `augment()`

### Module Structure
```
augmentation/    # Augmentation strategies (visual + textual)
dataset/         # Dataset implementations (all inherit BaseDataset)
training/        # OOP training pipeline (config.py, pipeline.py, trainer.py)
models/          # VQA models (SimpleVQA = ViT + BARTPho + classifier)
utils/           # compute_metrics, seed_everything, visualization
experiments/     # Shell scripts for reproducible experiments
```

## Critical Workflows

### Training Pipeline Entry Points
```bash
# Use train.py with CLI args (recommended)
python train.py --dataset_name vivqa --enable_augmentation --enable_curriculum --epochs 30

# Or use experiment scripts
./experiments/run_vivqa.sh  # Runs all 7 experiments
```

### Adding New Augmentation
1. Create class in `augmentation/visual/` or `augmentation/textual/`
2. Inherit from `BaseImageAugmentation` or `BaseTextAugmentation`
3. Implement `_configure_parameters(self)` - map `self.difficulty` (float 0.0-1.0) to augmentation params
4. Implement `augment(self, data)` method
5. Register in factory: `AugmentationFactory.register_image_augmentation('name', YourClass)`
6. Example pattern from `MaskedImageAugmentation`:
   ```python
   def _configure_parameters(self):
       self.mask_ratio = 0.15 + (self.difficulty * 0.60)  # 0.15 to 0.75
   ```

### Adding New Dataset
1. Create class in `dataset/` inheriting from `BaseDataset`
2. Implement `get_label_encoder()` returning `{answer: label_id}` dict
3. Implement `process_csv()` or `process_json()` returning dict with keys: `questions`, `answers`, `img_paths`
4. Add to `DatasetFactory.DATASET_CLASSES` in `training/pipeline.py`
5. Add default paths to `DataConfig._set_default_paths()` in `training/config.py`
6. Register download config in `data/config.yaml`

## Dataset-Specific Conventions

### Dataset Paths (Auto-Configured)
- **ViVQA**: CSV format, MSCOCO images (create via `scripts/create_vivqa_image.py`)
- **OpenViVQA**: JSON format, web-crawled images
- **ViTextVQA**: JSON format, text-rich images
- Default paths set in `DataConfig._set_default_paths()` - only override if non-standard

### Label Encoding
Each dataset implements `get_label_encoder()` differently:
- ViVQA: Simple answer set from CSV
- OpenViVQA: Top-N frequent answers
- Check existing datasets in `dataset/` for patterns

## Curriculum Learning System

### Difficulty Progression
- Scheduler returns **float 0.0-1.0** (NOT discrete levels)
- Strategies: `'linear'`, `'cosine'`, `'exponential'`, `'step'`, `'polynomial'`
- Example from experiments: 30 epochs = 9 easy + 9 medium + 12 hard (9/30=0.3, 18/30=0.6, 30/30=1.0)
- Augmentation classes map difficulty to params in `_configure_parameters()`

### Scheduler Usage
```python
from augmentation import CurriculumScheduler
scheduler = CurriculumScheduler(total_epochs=30, strategy='linear')
difficulty = scheduler.get_difficulty(epoch)  # Returns 0.0-1.0
```

## Configuration Management

### Never Use String Configs - Use Dataclasses
```python
# Programmatic configuration
from training import ExperimentConfig, ModelConfig, DataConfig, AugmentationConfig, TrainingConfig

config = ExperimentConfig(
    model=ModelConfig(vis_model_name='google/vit-base-patch16-224'),
    data=DataConfig(dataset_name='vivqa', batch_size=64),
    augmentation=AugmentationConfig(enable_augmentation=True, augmentation_type='masked'),
    training=TrainingConfig(epochs=30, learning_rate=1e-4)
)
```

### Default Model Choices
- **Vision**: `google/vit-base-patch16-224` (default), `microsoft/beit-base-patch16-224-pt22k-ft22k`
- **Text**: `vinai/bartpho-syllable-base` (default), `vinai/bartpho-syllable`, `FacebookAI/xlm-roberta-base`
- Model architecture: `SimpleVQA` = vision encoder + text encoder + concat + classifier

## Testing & Validation

### Running Experiments
- Use experiment scripts in `experiments/` for reproducibility
- Observations saved to `observations/{dataset}-{exp_name}-{seed}/epoch_XXX/`
- Each epoch saves: original/augmented images, questions, predictions, ground truth
- WandB integration: `--report_to_wandb --wandb_name PROJECT_NAME`

### Experiment Naming Convention
Format: `{dataset}-exp{N}_{description}-{seed}`
Example: `vivqa-exp7_full_augment_cl-42`

## Data Management

### Download Datasets
```bash
cd data && python download.py
# Enter indices: 1,2,3 (MSCOCO, ViVQA, OpenViVQA)
```

### Special Requirements
- **ViVQA**: Requires MSCOCO images + `scripts/create_vivqa_image.py` to generate dataset images
- **Git LFS**: Required for large file downloads
- **HuggingFace Token**: Set in `.env` (copy from `.env.example`)

### Dataset Config Format
See `data/config.yaml` for download configs - supports `direct`, `git`, `huggingface` download types

## Dependencies & Environment

### Critical Libraries
- PyTorch 2.9.0 (not compatible with older versions)
- Transformers 4.57.1 with `[torch]` extras
- Python 3.12.9 (strict requirement)

### Environment Setup
```bash
conda create -n vqa-template python=3.12.9 --y
conda activate vqa-template
pip install -r requirements.txt
```

## Common Patterns

### Augmentation Application
Augmentations passed as callables to dataset:
```python
image_aug = AugmentationFactory.create_image_augmentation('masked', difficulty=0.5)
dataset = DatasetFactory.create_dataset(
    dataset_name='vivqa',
    ann_path=train_ann_path,
    img_dir=train_img_dir,
    text_processor=tokenizer,
    vis_processor=processor,
    image_augmentation=image_aug.augment  # Pass method, not instance
)
```

### Visualization & Debugging
- BBox visualization: `scripts/visualize_bbox.py --host 0.0.0.0 --port 7860`
- Sample observation: `--enable_sample_observation` flag saves augmented samples per epoch
- Check `observations/` directory for training artifacts

## Documentation References
- Architecture diagram: `docs/ARCHITECTURE_DIAGRAM.md`
- Dataset comparison: `docs/DATASET_QUICK_REFERENCE.md`
- Augmentation guide: `augmentation/README.md`
- Training guide: `training/README.md`
