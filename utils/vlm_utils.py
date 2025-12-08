"""
Utility functions for VLM training and evaluation.
"""

import gc
import json
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


def set_global_seed(seed: int):
    """
    Set global seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clear_memory():
    """Clear CUDA memory and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_label_encoder(dataset_name: str) -> Dict[str, int]:
    """
    Get label encoder for a dataset.
    
    Args:
        dataset_name: Name of dataset ('ViVQA', 'OpenViVQA', 'ViVQA-X')
    
    Returns:
        Dictionary mapping answers to indices
    """
    if dataset_name == 'ViVQA':
        return get_label_encoder_vivqa()
    elif dataset_name == 'OpenViVQA':
        return get_label_encoder_openvivqa()
    elif dataset_name == 'ViVQA-X':
        return get_label_encoder_vivqax()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_label_encoder_vivqa() -> Dict[str, int]:
    """Get label encoder for ViVQA dataset."""
    train_data = pd.read_csv("data/vivqa/train.csv")
    test_data = pd.read_csv("data/vivqa/test.csv")
    answers = pd.concat([train_data["answer"], test_data["answer"]])
    sorted_answers = sorted(set(str(a) for a in answers.tolist()))
    return {answer: i for i, answer in enumerate(sorted_answers)}


def get_label_encoder_openvivqa() -> Dict[str, int]:
    """Get label encoder for OpenViVQA dataset."""
    with open("data/openvivqa/vlsp2023_train_data.json", 'r', encoding='utf-8') as f:
        train_anns = json.load(f)['annotations']
    with open("data/openvivqa/vlsp2023_dev_data.json", 'r', encoding='utf-8') as f:
        dev_anns = json.load(f)['annotations']
    
    answers = []
    for ann in train_anns.values():
        answers.append(str(ann['answer']))
    for ann in dev_anns.values():
        answers.append(str(ann['answer']))
    
    sorted_answers = sorted(set(answers))
    return {answer: i for i, answer in enumerate(sorted_answers)}


def get_label_encoder_vivqax() -> Dict[str, int]:
    """Get label encoder for ViVQA-X dataset."""
    with open("data/vivqa-x/ViVQA-X_train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open("data/vivqa-x/ViVQA-X_val.json", 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    answers = []
    for item in train_data:
        answers.append(str(item['answer']))
    for item in val_data:
        answers.append(str(item['answer']))
    
    sorted_answers = sorted(set(answers))
    return {answer: i for i, answer in enumerate(sorted_answers)}


def get_dataset_paths(dataset_name: str) -> Dict[str, str]:
    """
    Get paths for a dataset.
    
    Args:
        dataset_name: Name of dataset
    
    Returns:
        Dictionary with train_ann, val_ann, train_img_dir, val_img_dir
    """
    paths = {
        'ViVQA': {
            'train_ann': 'data/vivqa/train.csv',
            'val_ann': 'data/vivqa/test.csv',
            'train_img_dir': 'data/vivqa/images',
            'val_img_dir': 'data/vivqa/images',
        },
        'OpenViVQA': {
            'train_ann': 'data/openvivqa/vlsp2023_train_data.json',
            'val_ann': 'data/openvivqa/vlsp2023_dev_data.json',
            'train_img_dir': 'data/openvivqa/training-images',
            'val_img_dir': 'data/openvivqa/dev-images',
        },
        'ViVQA-X': {
            'train_ann': 'data/vivqa-x/ViVQA-X_train.json',
            'val_ann': 'data/vivqa-x/ViVQA-X_val.json',
            'train_img_dir': 'data/MSCOCO/train2014',
            'val_img_dir': 'data/MSCOCO/val2014',
        },
    }
    
    if dataset_name not in paths:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return paths[dataset_name]


def get_augmented_paths(dataset_name: str) -> str:
    """
    Get path to augmented dataset JSON file.
    
    Args:
        dataset_name: Name of dataset
    
    Returns:
        Path to augmented JSON file
    """
    augmented_paths = {
        'ViVQA': 'simple_augmented_datasets/vivqa/ViVQA_simple_augmented.json',
        'OpenViVQA': 'augmented_datasets/OpenViVQA_sap_augmented.json',
        'ViVQA-X': 'simple_augmented_datasets/vivqa-x/ViVQA-X_simple_augmented.json',
    }
    
    if dataset_name not in augmented_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return augmented_paths[dataset_name]


def save_evaluation_results(
    results: Dict,
    save_path: str,
    include_details: bool = True
):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results dict
        save_path: Path to save JSON file
        include_details: Include detailed per-sample results
    """
    output = {
        'accuracy': results['accuracy'],
        'correct': results['correct'],
        'total': results['total'],
    }
    
    if include_details and 'results' in results:
        output['details'] = results['results']
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Results saved to: {save_path}")


def print_evaluation_summary(results: Dict, title: str = "Evaluation Results"):
    """Print evaluation summary."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    
    if 'results' in results:
        # Show some examples
        print(f"\nSample predictions:")
        for i, r in enumerate(results['results'][:5]):
            status = "✓" if r['correct'] else "✗"
            print(f"  {status} Q: {r['question'][:50]}...")
            print(f"      Pred: {r['prediction']}, GT: {r['ground_truth']}")
