# VQA Template

A comprehensive Vietnamese Visual Question Answering (VQA) dataset template and research framework.

## 📚 Documentation

**NEW**: Comprehensive [Vietnamese VQA Dataset Survey](docs/README.md) covering:

- Detailed analysis of 6 Vietnamese VQA datasets (ViVQA, OpenViVQA, ViTextVQA, ViOCRVQA, ViCLEVR, EVJVQA)
- Dataset building methodologies and generation approaches
- Quality assessments and improvement recommendations
- 100+ template examples for dataset enhancement
- Contribution guidelines

👉 **[Read the Full Documentation](docs/README.md)** to understand how these datasets were built and how to contribute improvements.

## 🚀 Quick Start

## Requirements

- Git LFS - [Installation-Linux](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md)
- Python 3.12.9
- Setup .env file with HuggingFace credentials:

```bash
cp .env.example .env
# Edit .env and set HF_USER and HF_TOKEN
```

### Installation

```bash
# Create conda environment
conda create -n vqa-template python=3.12.9 --y
conda activate vqa-template

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

```bash
cd data
python download.py
# Enter the dataset index to download (eg. 1,2,3 to download MSCOCO, ViVQA and OpenViVQA dataset)
```

### Generate COCO Images for ViVQA

```bash
cd scripts
python create_vivqa_image.py
```

### Tools

- Download object detection features

```bash
cd ./data/obj36_feat
python download.py
```

- Visulize bbox

```bash
python scripts/visualize_bbox.py
```

## Template Extraction & Hard Sample Mining

Surface under-covered question patterns before training to target augmentation where it matters most:

```bash
python -m examples.template_extraction_usage
```

The script loads the ViVQA dataset, extracts reusable question templates, scores sample difficulty (0.0-1.0), and writes `runs/template_analysis/vivqa_hard_samples.json`. Feed that payload into your augmentation jobs to upsample rare templates, rare answers, or long/complex questions.

## Training

### Run All Experiments

```bash
cd /home/thuannd/Repository/VQA_Template
./experiments/run_vivqa.sh
```

### 📊 Experiments Included

| # | Name | Text Aug | Image Aug | Curriculum | Command |
|---|------|----------|-----------|------------|---------|
| 1 | Baseline | ❌ | ❌ | ❌ | `exp1_baseline` |
| 2 | Text Only | ✅ | ❌ | ❌ | `exp2_text_augment` |
| 3 | Image Only | ❌ | ✅ | ❌ | `exp3_image_augment` |
| 4 | Text + Image | ✅ | ✅ | ❌ | `exp4_text_image_augment` |
| 5 | Text + CL | ✅ | ❌ | ✅ | `exp5_text_augment_cl` |
| 6 | Image + CL | ❌ | ✅ | ✅ | `exp6_image_augment_cl` |
| 7 | Full (⭐ Best) | ✅ | ✅ | ✅ | `exp7_full_augment_cl` |

### Basic Training

Train the VQA model with default settings:

```bash
python train.py \
    --dataset_name vivqa \
    --batch_size 64 \
    --epochs 30 \
    --learning_rate 1e-4
```

### Training with Augmentation

```bash
python train.py \
    --dataset_name vivqa \
    --enable_augmentation \
    --augmentation_type masked \
    --patch_size 16 \
    --epochs 30
```

### Training with Curriculum Learning ⭐ (Recommended)

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

### Key Features

✅ **Curriculum Learning** - Progressive difficulty (EASY → MEDIUM → HARD)
✅ **Dynamic Augmentation** - Automatically adjusts based on training epoch
✅ **Type-Safe Configuration** - Dataclasses for all configurations
✅ **Extensible Architecture** - Easy to add new augmentation strategies
✅ **WandB Integration** - Track experiments with Weights & Biases

### Programmatic Usage

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

### Custom Augmentation

```python
from augmentation import BaseImageAugmentation, AugmentationFactory
from PIL import Image, ImageFilter

class MyCustomAugmentation(BaseImageAugmentation):
    def _configure_parameters(self):
        # Configure based on difficulty (0.0-1.0)
        self.strength = self.difficulty  # Direct mapping
        # Or custom mapping:
        # self.strength = 0.1 + (self.difficulty * 0.9)  # Range: 0.1 to 1.0
    
    def augment(self, image: Image.Image, **kwargs) -> Image.Image:
        # Your augmentation logic here
        return image
    
    def get_augmentation_info(self) -> dict:
        return {
            'type': 'MyCustom',
            'difficulty': self.difficulty,
            'strength': self.strength
        }

# Register and use
AugmentationFactory.register_image_augmentation('my_custom', MyCustomAugmentation)

augmentor = AugmentationFactory.create_image_augmentation(
    augmentation_type='my_custom',
    difficulty=0.5  # Float value between 0.0 and 1.0
)
```

### Available Command-Line Arguments

#### Model Arguments

- `--vis_model_name` - Vision model (default: google/vit-base-patch16-224)
- `--text_model_name` - Text model (default: vinai/bartpho-syllable-base)

#### Dataset Arguments

- `--dataset_name` - Dataset to use (vivqa, openvivqa, vitextvqa, etc.)
- `--batch_size` - Batch size (default: 64)
- `--seq_len` - Sequence length (default: 64)

#### Training Arguments

- `--epochs` - Number of epochs (default: 30)
- `--learning_rate` - Learning rate (default: 1e-4)
- `--weight_decay` - Weight decay (default: 1e-4)
- `--fp16` - Enable mixed precision training

#### Augmentation Arguments

- `--enable_augmentation` - Enable image augmentation
- `--augmentation_type` - Type of augmentation (masked, none)
- `--patch_size` - Patch size for masked augmentation (default: 16)

#### Curriculum Learning Arguments

- `--enable_curriculum` - Enable curriculum learning
- `--easy_epochs` - Number of easy epochs
- `--medium_epochs` - Number of medium epochs
- `--hard_epochs` - Number of hard epochs

#### Logging Arguments

- `--report_to_wandb` - Enable WandB logging
- `--wandb_name` - WandB project name
- `--run_name` - Run name for identification
- `--output_dir` - Output directory (default: runs)

### Experiment Tracking with Weights & Biases

This project supports [Weights & Biases (wandb)](https://docs.wandb.ai/) for experiment tracking, visualization, and model management.

#### Setup wandb

1. Create a free account at [wandb.ai](https://wandb.ai/)
2. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)
3. Copy `.env.example` to `.env` and add your API key:

   ```bash
   cp .env.example .env
   # Edit .env and set WANDB_API_KEY=your_api_key_here
   ```

4. Alternatively, login via command line:

   ```bash
   wandb login
   ```

#### Training with wandb

Enable wandb logging by adding the `--report_to_wandb` flag:

```bash
python train.py \
    --dataset_name vivqa \
    --batch_size 64 \
    --epochs 30 \
    --learning_rate 1e-4 \
    --report_to_wandb \
    --wandb_name "VQA-Template" \
    --run_name "baseline"
```

#### wandb Features

When enabled, the following metrics and information are automatically logged:

- **Training metrics**: loss, learning rate per step
- **Validation metrics**: loss, accuracy per epoch
- **Hyperparameters**: all model and training configurations
- **System metrics**: GPU/CPU usage, memory
- **Model checkpoints**: saved at each evaluation

#### View Results

After training starts, you'll see a link to your wandb dashboard where you can:

- Monitor training progress in real-time
- Compare different experiments
- Visualize metrics with interactive charts
- Share results with your team

Example wandb dashboard URL: `https://wandb.ai/<your-username>/VQA-Template/runs/<run-id>`

#### Training Arguments for wandb

- `--report_to_wandb`: Enable wandb logging (default: disabled)
- `--wandb_name`: Project name in wandb (default: "VQA-Template")
- `--run_name`: Name for this specific run (default: "run")

The final run will be named as: `{dataset_name}-{run_name}-{seed}`

Example: `ViVQA-baseline-71`

## Tips & Best Practices

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

## Dataset

For a comprehensive analysis of Vietnamese VQA datasets, see our [Dataset Survey Documentation](docs/README.md):

- 📊 [Complete Dataset Survey](docs/DATASET_SURVEY.md) - Detailed analysis of building methods, generation approaches, and quality
- ⚡ [Quick Reference Guide](docs/DATASET_QUICK_REFERENCE.md) - Dataset comparison and key findings
- 🔧 [Template Improvements](docs/TEMPLATE_IMPROVEMENTS.md) - 100+ concrete template examples
- 🤝 [Contributing Guidelines](docs/CONTRIBUTING.md) - How to contribute improvements

### Supported Datasets

- [ ] [MSCOCO](https://cocodataset.org/#download) - Microsoft Common Objects in Context
- [ ] [ViVQA](https://github.com/kh4nh12/ViVQA) - Vietnamese Visual Question Answering Dataset
- [ ] [OpenViVQA](https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset) - Open Domain Vietnamese Visual Question Answering Dataset
- [ ] [ViVQA-X](https://huggingface.co/datasets/VLAI-AIVN/ViVQA-X) - Vietnamese VQA with Natural Language Explanations
- [ ] [ViOCR-VQA](https://huggingface.co/datasets/VLAI-AIVN/ViOCR-VQA) - Vietnamese Visual Question Answering Dataset with OCR
- [ ] [EVJ-VQA](https://huggingface.co/datasets/dinhanhx/evjvqa) - Vietnamese Visual Question Answering Dataset for e-commerce domain
- [ ] [ViTextVQA](https://huggingface.co/datasets/minhquan6203/ViTextVQA) - Vietnamese Text-based Visual Question Answering Dataset
- [ ] [ViCLEVR](docs/DATASET_SURVEY.md#viclevr) - Vietnamese CLEVR Dataset (planned/placeholder)

## Paper

- [ ] [Data Augmentation for Visual Question Answering](https://aclanthology.org/W17-3529.pdf) - Implemented as Vietnamese rule-based augmentation
- [ ] [Discovering the Unknown Knowns: Turning Implicit Knowledge in the Dataset into Explicit Training Examples for Visual Question Answering](https://arxiv.org/abs/2109.06122)
- [ ] [Stacked Attention Networks for Image Question Answering](https://arxiv.org/pdf/1511.02274)
- [ ] [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490)
- [ ] [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)

## Contributing

We welcome contributions to improve Vietnamese VQA datasets and tools! See our [Contributing Guidelines](docs/CONTRIBUTING.md) for:

- Priority contribution areas
- Template enhancement opportunities
- Code contribution workflow
- Documentation improvements

**Top Contribution Opportunities**:

1. ✨ Expand ViVQA templates (35+ new templates)
2. 🛒 Create EVJVQA Vietnamese e-commerce templates
3. 🔬 Integrate ViCLEVR dataset support
4. 🤖 Build modern LLM-based generation pipeline
5. 📊 Develop quality analysis tools

## Citation

If you use this repository or the dataset survey in your research, please cite:

```bibtex
@misc{vqa_template_2024,
  title={Vietnamese VQA Template: Datasets, Analysis, and Tools},
  author={VQA Template Team},
  year={2024},
  howpublished={\url{https://github.com/ThuanNaN/VQA_Template}},
  note={Framework and comprehensive survey for Vietnamese Visual Question Answering}
}
```
