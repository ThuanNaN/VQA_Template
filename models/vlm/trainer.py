"""
VQA Trainer and evaluation utilities for VLM.
"""

import gc
import time
from typing import Dict, List, Optional, Any, Callable

import torch
from tqdm import tqdm

from trl import SFTTrainer


class VQACollator:
    """
    Data collator for VQA task with VLMs.
    Handles conversation formatting and image processing.
    """
    
    def __init__(self, processor):
        """
        Args:
            processor: VLM processor (e.g., Qwen2VLProcessor)
        """
        self.processor = processor
    
    def __call__(self, examples: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of conversation examples.
        
        Args:
            examples: List of conversations, each is a list of message dicts
        
        Returns:
            Batch dict with input_ids, attention_mask, pixel_values, labels
        """
        # Apply chat template to each conversation
        texts = [
            self.processor.apply_chat_template(example, tokenize=False)
            for example in examples
        ]
        
        # Extract images from user messages
        image_inputs = [
            example[1]["content"][0]["image"]  # user message, first content item (image)
            for example in examples
        ]
        
        # Process batch
        batch = self.processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True
        )
        
        # Create labels (mask padding tokens)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(0.5)


def evaluate_vqa_accuracy(
    model,
    processor,
    eval_dataset,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    max_new_tokens: int = 32,
    show_progress: bool = True,
    clear_memory_every: int = 50,
) -> Dict[str, Any]:
    """
    Evaluate VQA accuracy by generating answers and comparing with ground truth.
    
    Args:
        model: The VLM model
        processor: The VLM processor
        eval_dataset: Evaluation dataset (VLMVQADataset)
        device: Device to run on
        max_samples: Maximum number of samples to evaluate (None = all)
        max_new_tokens: Maximum tokens to generate
        show_progress: Show progress bar
        clear_memory_every: Clear GPU memory every N samples
        
    Returns:
        Dictionary with:
            - accuracy: float
            - correct: int
            - total: int
            - predictions: List[str]
            - ground_truths: List[str]
            - results: List[Dict] with detailed per-sample results
    """
    model.eval()
    
    correct = 0
    total = 0
    predictions = []
    ground_truths = []
    detailed_results = []
    
    num_samples = len(eval_dataset) if max_samples is None else min(max_samples, len(eval_dataset))
    
    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Evaluating VQA accuracy")
    
    with torch.no_grad():
        for idx in iterator:
            try:
                # Get conversation and raw item
                conversation = eval_dataset[idx]
                raw_item = eval_dataset.get_raw_item(idx)
                
                # Prepare input (system + user only, no assistant)
                text = processor.apply_chat_template(
                    conversation[0:2],
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Get image
                image = conversation[1]["content"][0]["image"]
                
                # Process inputs
                inputs = processor(
                    text=[text],
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id
                )
                
                # Decode
                generated_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
                
                # Extract answer from generated text
                pred_answer = _extract_answer(generated_text)
                gt_answer = raw_item['answer'].strip().lower()
                
                predictions.append(pred_answer)
                ground_truths.append(gt_answer)
                
                # Check exact match
                is_correct = pred_answer == gt_answer
                if is_correct:
                    correct += 1
                total += 1
                
                # Store detailed result
                detailed_results.append({
                    'idx': idx,
                    'question': raw_item['question'],
                    'prediction': pred_answer,
                    'ground_truth': gt_answer,
                    'correct': is_correct,
                    'img_path': raw_item['img_path']
                })
                
                # Clear memory periodically
                del inputs, generated_ids
                if idx % clear_memory_every == 0:
                    clear_memory()
                    
            except Exception as e:
                print(f"Error evaluating sample {idx}: {e}")
                predictions.append("")
                ground_truths.append(raw_item['answer'].strip().lower())
                total += 1
                continue
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions,
        "ground_truths": ground_truths,
        "results": detailed_results
    }


def _extract_answer(generated_text: str) -> str:
    """
    Extract the answer from generated text.
    Handles different response formats.
    """
    text = generated_text.strip()
    
    # Try to extract after "assistant" marker
    if "assistant" in text.lower():
        parts = text.lower().split("assistant")
        text = parts[-1].strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "answer:",
        "the answer is",
        "câu trả lời:",
        "câu trả lời là",
        "đáp án:",
        "đáp án là",
    ]
    
    text_lower = text.lower()
    for prefix in prefixes_to_remove:
        if text_lower.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    
    # Clean up
    text = text.strip().lower()
    
    # Remove quotes if present
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    
    return text


def compute_vqa_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """
    Compute VQA metrics.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
    
    Returns:
        Dictionary with metrics (accuracy, etc.)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    if len(predictions) == 0:
        return {"accuracy": 0.0}
    
    # Exact match accuracy
    correct = sum(
        1 for pred, gt in zip(predictions, ground_truths)
        if pred.strip().lower() == gt.strip().lower()
    )
    accuracy = correct / len(predictions)
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(predictions)
    }
