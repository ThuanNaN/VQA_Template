import pandas as pd
import json
from .base import BaseDataset

class OpenViVQADataset(BaseDataset):
    train_ann = "data/openvivqa/vlsp2023_train_data.json"
    def __init__(self, ann_path, img_dir, text_processor, vis_processor):
        super().__init__(ann_path, img_dir, text_processor, vis_processor)

    def get_label_encoder(self):
        with open(OpenViVQADataset.train_ann, "r") as f:
            data = json.load(f)
        keys_id = list(data['annotations'].keys())
        answers = []
        for idx in keys_id:
            answers.append(data['annotations'][str(idx)]['answer'])
        sorted_answers = sorted(set(answers))
        return {answer: i for i, answer in enumerate(sorted_answers)}
    
