# VQA Template

## Requirements

- Git LFS - [Installation-Linux](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md)
- Python 3.12.9

## Install dependencies

```bash
conda create -n vqa-template python=3.12.9 --y
conda activate vqa-template
pip3 install -r requirements.txt
```

Download dataset

```bash
cd data
python download.py
# Enter the dataset index to download (eg. 1,2,3 to download MSCOCO, ViVQA and OpenViVQA dataset)
```

Generate COCO images for `vivqa` dataset

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
python vis_bbox.py --host 0.0.0.0 --port 7860
```

## Training

### Basic Training

Train the VQA model with default settings:

```bash
python train.py \
    --dataset_name ViVQA \
    --batch_size 64 \
    --epochs 30 \
    --learning_rate 1e-4
```

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
    --dataset_name ViVQA \
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

## Dataset

- [ ] [MSCOCO](https://cocodataset.org/#download) - Microsoft Common Objects in Context
- [ ] [ViVQA](https://github.com/kh4nh12/ViVQA) - Vietnamese Visual Question Answering Dataset
- [ ] [OpenViVQA](https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset) - Open Domain Vietnamese Visual Question Answering Dataset
- [ ] [ViVQA-X](https://huggingface.co/datasets/VLAI-AIVN/ViVQA-X) - Vietnamese VQA with Natural Language Explanations
- [ ] [ViOCR-VQA](https://huggingface.co/datasets/VLAI-AIVN/ViOCR-VQA) - Vietnamese Visual Question Answering Dataset with OCR
- [ ] [EVJ-VQA](https://huggingface.co/datasets/dinhanhx/evjvqa) - Vietnamese Visual Question Answering Dataset for e-commerce domain
- [ ] [ViTextVQA](https://huggingface.co/datasets/minhquan6203/ViTextVQA) - Vietnamese Text-based Visual Question Answering Dataset

## Paper

- [x] [Data Augmentation for Visual Question Answering](https://aclanthology.org/W17-3529.pdf) - Implemented as Vietnamese rule-based augmentation
- [ ] [Discovering the Unknown Knowns: Turning Implicit Knowledge in the Dataset into Explicit Training Examples for Visual Question Answering](https://arxiv.org/abs/2109.06122)
- [ ] [Stacked Attention Networks for Image Question Answering](https://arxiv.org/pdf/1511.02274)
- [ ] [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490)
- [ ] [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)
