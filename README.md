# VQA Template

## Requirements

- Git LFS - [Installation-Linux](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md)
- Python 3.12.9

## Install dependencies

Python dependencies are managed by [uv](https://github.com/astral-sh/uv)

```bash
conda create -n vqa-template python=3.12.9 --y
conda activate vqa-template
pip install uv
uv pip install -r requirements.txt
```

Download dataset

```bash
cd data
python download.py
# Enter the dataset index to download (eg. 1,2,3) to download MSCOCO, ViVQA and OpenViVQA dataset
```

Generate COCO images for `vivqq` dataset

```bash
cd scripts
python vivqa_image.py
```

## Dataset

- [ ] [ViVQA](https://github.com/kh4nh12/ViVQA)
- [ ] [OpenViVQA](https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset)

## Paper

- [ ] [Data Augmentation for Visual Question Answering](https://aclanthology.org/W17-3529.pdf)
- [ ] [Discovering the Unknown Knowns: Turning Implicit Knowledge in the Dataset into Explicit Training Examples for Visual Question Answering](https://arxiv.org/abs/2109.06122)
- [ ] [Stacked Attention Networks for Image Question Answering](https://arxiv.org/pdf/1511.02274)
- [ ] [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490)
- [ ] [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)
