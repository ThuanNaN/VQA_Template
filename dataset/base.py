import os
import json
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class BaseDataset(Dataset):
    def __init__(self, ann_path, img_dir, 
                 text_processor, vis_processor,
                 **kwargs):
        super(BaseDataset, self).__init__()
        self.img_dir = img_dir
        self.text_processor = text_processor
        self.vis_processor = vis_processor
        self.label_encoder = self.get_label_encoder()
        self.data = self.load_data(ann_path)
        self.kwargs = kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data['img_paths'][idx]
        pil_image = Image.open(img_path).convert('RGB')
        image = self.vis_processor(pil_image, return_tensors="pt")
        question = self.data['questions'][idx]
        answer = self.data['answers'][idx]
        question = self.text_processor(question, return_tensors="pt", **self.kwargs)
        answer = self.label_encoder[answer]
        return image, question, answer, 

    def load_data(self, ann_path):
        if ann_path.endswith('.csv'):
            data = self.process_csv(ann_path)
        elif ann_path.endswith('.json'):
            data = self.process_json(ann_path)
        else:
            raise ValueError(f"Unsupported file format for {ann_path}")
        return data

    def process_csv(self, ann_path) -> dict:
        data = pd.read_csv(ann_path)
        questions = []
        answers = []
        img_paths = []
        for _, item in data.iterrows():
            questions.append(item['question'])
            answers.append(item['answer'])
            img_id = str(item['img_id']).zfill(12)
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            img_paths.append(img_path)
        return {
            'questions': questions,
            'answers': answers,
            'img_paths': img_paths
        }

    def process_json(self, ann_path) -> dict:
        with open(ann_path, 'r') as f:
            anns = json.load(f)
        image_anns = anns['images']
        iqa_anns = anns['annotations']
        keys_id = list(iqa_anns.keys())
        questions = []
        answers = []
        img_paths = []
        for idx in keys_id:
            iqa = iqa_anns[idx]
            img_id = iqa['image_id']
            img_path = os.path.join(self.img_dir, image_anns[str(img_id)])
            questions.append(iqa['question'])
            answers.append(iqa['answer'])
            img_paths.append(img_path)
        return {
            'questions': questions,
            'answers': answers,
            'img_paths': img_paths
        }

    def get_label_encoder(self):
        raise NotImplementedError