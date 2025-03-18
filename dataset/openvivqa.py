import pandas as pd
import json
from .base import BaseDataset

class OpenViVQADataset(BaseDataset):
    train_ann = "data/openvivqa/vlsp2023_train_data.json"
    dev_ann = "data/openvivqa/vlsp2023_dev_data.json"

    def __init__(self, ann_path, img_dir, text_processor, vis_processor):
        super().__init__(ann_path, img_dir, text_processor, vis_processor)

    def get_label_encoder(self):
        with open(OpenViVQADataset.train_ann, "r") as f:
            train_data = json.load(f)
        train_keys_id = list(train_data['annotations'].keys())

        with open(OpenViVQADataset.dev_ann, "r") as f:
            dev_data = json.load(f)
        dev_keys_id = list(dev_data['annotations'].keys())

        answers = []
        for train_idx, dev_idx in zip(train_keys_id, dev_keys_id):
            answers.append(train_data['annotations'][str(train_idx)]['answer'])
            answers.append(dev_data['annotations'][str(dev_idx)]['answer'])

        sorted_answers = sorted(set(answers))
        return {answer: i for i, answer in enumerate(sorted_answers)}
    