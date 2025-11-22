from typing import List, Dict, Union
import sys
import csv
import base64
import time
import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)


csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_obj_tsv(fname, topk=None) -> List[Dict[str, Union[str, int, np.ndarray]]]:
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    logger.info("Start to load Faster-RCNN detected objects from %s", fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    logger.info("Loaded %d images in file %s in %d seconds.", len(data), fname, elapsed_time)
    return data


def ds_collate_fn(batch):
    image_output = {}
    question_output = {}
    answer_output = []

    pixel_values = []
    question_input_ids = []
    question_attention_mask = []
    question_token_type_ids = []
    for image, question, answer in batch:
        pixel_values.append(image["pixel_values"].squeeze(0))
        question_input_ids.append(question["input_ids"].squeeze(0))
        question_attention_mask.append(question["attention_mask"].squeeze(0))
        question_token_type_ids.append(question["token_type_ids"].squeeze(0))
        answer_output.append(answer)
    image_output["pixel_values"] = torch.stack(pixel_values)
    question_output["input_ids"] = torch.stack(question_input_ids)
    question_output["attention_mask"] = torch.stack(question_attention_mask)
    question_output["token_type_ids"] = torch.stack(question_token_type_ids)
    answer_output = torch.tensor(answer_output)
    return image_output, question_output, answer_output


def dict2device(data, device):
    for k, v in data.items():
        data[k] = v.to(device)
    return data
