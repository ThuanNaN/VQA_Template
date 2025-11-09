import pandas as pd
from .base import BaseDataset

class ViVQADataset(BaseDataset):
    train_ann = "data/vivqa/train.csv"
    test_ann = "data/vivqa/test.csv"

    def __init__(self, ann_path, img_dir, text_processor, vis_processor, **kwargs):
        super().__init__(ann_path, img_dir, text_processor, vis_processor, **kwargs)

    def get_label_encoder(self):
        with open(ViVQADataset.train_ann, "r") as f:
            train_data = pd.read_csv(f)
        with open(ViVQADataset.test_ann, "r") as f:
            test_data = pd.read_csv(f)
        train_answers = train_data["answer"]
        test_answers = test_data["answer"]
        answers = pd.concat([train_answers, test_answers])
        sorted_answers = sorted(set(answers.tolist()))
        return {answer: i for i, answer in enumerate(sorted_answers)}
    
