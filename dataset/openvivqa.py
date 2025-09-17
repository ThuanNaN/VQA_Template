import os
import json
from .base import BaseDataset

class OpenViVQADataset(BaseDataset):
    train_ann = "data/openvivqa/vlsp2023_train_data.json"
    dev_ann = "data/openvivqa/vlsp2023_dev_data.json"

    def __init__(self, ann_path, img_dir, text_processor, vis_processor, **kwargs):
        super().__init__(ann_path, img_dir, text_processor, vis_processor, **kwargs)

    def get_label_encoder(self):
        with open(OpenViVQADataset.train_ann, "r") as f:
            train_data = json.load(f)
        train_keys_id = list(train_data['annotations'].keys())

        with open(OpenViVQADataset.dev_ann, "r") as f:
            dev_data = json.load(f)
        dev_keys_id = list(dev_data['annotations'].keys())

        answers = []
        for train_idx in train_keys_id:
            train_idx_question = train_data['annotations'][str(train_idx)]['question']
            train_idx_answer = train_data['annotations'][str(train_idx)]['answer']
            train_idx_answer = self.remove_continuous_sequences(train_idx_question, train_idx_answer)
            answers.append(train_idx_answer)

        for dev_idx in dev_keys_id:
            dev_idx_question = dev_data['annotations'][str(dev_idx)]['question']
            dev_idx_answer = dev_data['annotations'][str(dev_idx)]['answer']
            dev_idx_answer = self.remove_continuous_sequences(dev_idx_question, dev_idx_answer)
            answers.append(dev_idx_answer)

        sorted_answers = sorted(set(answers))
        return {answer: i for i, answer in enumerate(sorted_answers)}


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
            answer = self.remove_continuous_sequences(iqa['question'], iqa['answer'])
            answers.append(answer)
            img_paths.append(img_path)
        return {
            'questions': questions,
            'answers': answers,
            'img_paths': img_paths
        }

    def remove_continuous_sequences(self, question: str, answer: str) -> str:
        # Split the question and answer into words
        question_words = question.split()

        # Generate all possible continuous sequences from the question
        # Only consider sequences of 3+ words to avoid removing important semantic content
        sequences = set()
        for i in range(len(question_words)):
            for j in range(i + 1, len(question_words) + 1):
                sequence = ' '.join(question_words[i:j])
                # Only consider sequences of 3+ words for removal
                if len(question_words[i:j]) >= 3:
                    sequences.add(sequence)

        # Remove sequences from the answer that match any sequence in the question
        new_answer = answer
        for seq in sorted(sequences, key=len, reverse=True):
            if seq in new_answer:
                new_answer = new_answer.replace(seq, '').strip()
        
        return new_answer
    