from .base import BaseDataset
import os 
import json

class ViVQAXDataset(BaseDataset):
    train_ann = "data/vivqa-x/ViVQA-X_train.json"
    test_ann = "data/vivqa-x/ViVQA-X_val.json"

    def __init__(self, ann_path, img_dir, text_processor, vis_processor, **kwargs):
        super().__init__(ann_path, img_dir, text_processor, vis_processor, **kwargs)

    def get_label_encoder(self):
        with open(ViVQAXDataset.train_ann, "r") as f:
            train_data = json.load(f)
        with open(ViVQAXDataset.test_ann, "r") as f:
            test_data = json.load(f)

        answers = []
        for iqa in train_data:
            print()
            train_idx_answer = iqa['answer']
            answers.append(train_idx_answer)

        for iqa in test_data:
            test_idx_answer = iqa['answer']
            answers.append(test_idx_answer)

        sorted_answers = sorted(set(answers))
        return {answer: i for i, answer in enumerate(sorted_answers)}

    def process_json(self, ann_path) -> dict:
        with open(ann_path, 'r') as f:
            anns = json.load(f)
        questions = []
        answers = []
        img_paths = []
        for iqa in anns:
                img_path = os.path.join(self.img_dir, str(iqa['image_name']) if 'image_name' in iqa else str(iqa['img_id']))
                questions.append(iqa['question'])
                answers.append(iqa['answer'])
                img_paths.append(img_path)

        return {
            'questions': questions,
            'answers': answers,
            'img_paths': img_paths
        }


