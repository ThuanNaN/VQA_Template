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

## Dataset

For a comprehensive analysis of Vietnamese VQA datasets, including their construction methodologies, generation approaches, and quality assessments, see [Dataset Survey Documentation](docs/DATASET_SURVEY.md).

- [ ] [MSCOCO](https://cocodataset.org/#download) - Microsoft Common Objects in Context
- [ ] [ViVQA](https://github.com/kh4nh12/ViVQA) - Vietnamese Visual Question Answering Dataset
- [ ] [OpenViVQA](https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset) - Open Domain Vietnamese Visual Question Answering Dataset
- [ ] [ViVQA-X](https://huggingface.co/datasets/VLAI-AIVN/ViVQA-X) - Vietnamese VQA with Natural Language Explanations
- [ ] [ViOCR-VQA](https://huggingface.co/datasets/VLAI-AIVN/ViOCR-VQA) - Vietnamese Visual Question Answering Dataset with OCR
- [ ] [EVJ-VQA](https://huggingface.co/datasets/dinhanhx/evjvqa) - Vietnamese Visual Question Answering Dataset for e-commerce domain
- [ ] [ViTextVQA](https://huggingface.co/datasets/minhquan6203/ViTextVQA) - Vietnamese Text-based Visual Question Answering Dataset
- [ ] [ViCLEVR](docs/DATASET_SURVEY.md#viclevr) - Vietnamese CLEVR Dataset (planned/placeholder)

## Paper

- [ ] [Data Augmentation for Visual Question Answering](https://aclanthology.org/W17-3529.pdf)
- [ ] [Discovering the Unknown Knowns: Turning Implicit Knowledge in the Dataset into Explicit Training Examples for Visual Question Answering](https://arxiv.org/abs/2109.06122)
- [ ] [Stacked Attention Networks for Image Question Answering](https://arxiv.org/pdf/1511.02274)
- [ ] [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490)
- [ ] [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)
