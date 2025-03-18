import pandas as pd
from .base import BaseDataset

class ViVQADataset(BaseDataset):
    train_ann = "data/vivqa/train.csv"
    def __init__(self, ann_path, img_dir, text_processor, vis_processor):
        super().__init__(ann_path, img_dir, text_processor, vis_processor)

    def get_label_encoder(self):
        with open(ViVQADataset.train_ann, "r") as f:
            data = pd.read_csv(f)
        answers = data["answer"]
        sorted_answers = sorted(set(answers.tolist()))
        return {answer: i for i, answer in enumerate(sorted_answers)}
    
