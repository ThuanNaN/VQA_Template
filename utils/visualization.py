"""
Utilities for saving training observations and visualizations.

This module provides functions to save sample data during training,
including original and augmented images, questions, model predictions,
and ground truth answers.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image, ImageDraw, ImageFont
import torch
import logging

logger = logging.getLogger(__name__)


class SampleObserver:
    """
    Handles saving sample observations during training.
    
    For each epoch, saves:
    - Original and augmented images
    - Original and augmented questions
    - Model predictions
    - Ground truth answers
    """
    
    def __init__(self, save_dir: str, num_samples: int = 5):
        """
        Initialize the sample observer.
        
        Args:
            save_dir: Directory to save observations
            num_samples: Number of samples to save per epoch
        """
        self.save_dir = Path(save_dir)
        self.num_samples = num_samples
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SampleObserver initialized: saving {num_samples} samples to {save_dir}")
    
    def save_epoch_samples(
        self,
        epoch: int,
        samples: List[Dict[str, Any]],
        label_decoder: Dict[int, str]
    ):
        """
        Save sample observations for an epoch.
        
        Args:
            epoch: Current epoch number
            samples: List of sample dictionaries containing:
                - original_image: PIL Image or None
                - augmented_image: PIL Image or None
                - original_question: str or None
                - augmented_question: str or None
                - prediction: int (label index)
                - ground_truth: int (label index)
                - img_path: str (original image path)
            label_decoder: Dictionary mapping label indices to answer strings
        """
        epoch_dir = self.save_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(exist_ok=True)
        
        # Save metadata in JSON
        metadata = []
        
        for idx, sample in enumerate(samples[:self.num_samples]):
            sample_dir = epoch_dir / f"sample_{idx:03d}"
            sample_dir.mkdir(exist_ok=True)
            
            # Save images
            if sample.get('original_image') is not None:
                original_img_path = sample_dir / "image_original.jpg"
                sample['original_image'].save(original_img_path)
            
            if sample.get('augmented_image') is not None:
                augmented_img_path = sample_dir / "image_augmented.jpg"
                sample['augmented_image'].save(augmented_img_path)
            
            # Create comparison visualization if both images exist
            if sample.get('original_image') and sample.get('augmented_image'):
                comparison_path = sample_dir / "image_comparison.jpg"
                self._create_comparison_image(
                    sample['original_image'],
                    sample['augmented_image'],
                    comparison_path
                )
            
            # Decode predictions and ground truth
            pred_answer = label_decoder.get(sample['prediction'], f"unknown_{sample['prediction']}")
            gt_answer = label_decoder.get(sample['ground_truth'], f"unknown_{sample['ground_truth']}")
            
            # Prepare metadata entry
            sample_meta = {
                'sample_id': idx,
                'original_image_path': sample.get('img_path', ''),
                'original_question': sample.get('original_question'),
                'augmented_question': sample.get('augmented_question'),
                'model_prediction': pred_answer,
                'ground_truth': gt_answer,
                'is_correct': sample['prediction'] == sample['ground_truth'],
                'has_image_augmentation': sample.get('augmented_image') is not None,
                'has_text_augmentation': (
                    sample.get('original_question') != sample.get('augmented_question')
                    if sample.get('original_question') and sample.get('augmented_question')
                    else False
                )
            }
            
            metadata.append(sample_meta)
            
            # Save individual sample metadata
            with open(sample_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(sample_meta, f, ensure_ascii=False, indent=2)
            
            # Create text summary file
            self._create_text_summary(sample_dir / "summary.txt", sample_meta)
        
        # Save epoch-level metadata
        epoch_metadata = {
            'epoch': epoch,
            'num_samples': len(samples[:self.num_samples]),
            'samples': metadata
        }
        
        with open(epoch_dir / "epoch_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(epoch_metadata, f, ensure_ascii=False, indent=2)
        
        # Create epoch summary
        correct_samples = sum(1 for m in metadata if m['is_correct'])
        accuracy = correct_samples / len(metadata) if metadata else 0.0
        
        summary_text = f"Epoch {epoch} Sample Summary\n"
        summary_text += "=" * 50 + "\n\n"
        summary_text += f"Total samples: {len(metadata)}\n"
        summary_text += f"Correct predictions: {correct_samples}/{len(metadata)} ({accuracy:.2%})\n\n"
        
        for meta in metadata:
            summary_text += f"Sample {meta['sample_id']}:\n"
            summary_text += f"  Question (original): {meta['original_question']}\n"
            if meta['has_text_augmentation']:
                summary_text += f"  Question (augmented): {meta['augmented_question']}\n"
            summary_text += f"  Ground truth: {meta['ground_truth']}\n"
            summary_text += f"  Prediction: {meta['model_prediction']}\n"
            summary_text += f"  Correct: {'✓' if meta['is_correct'] else '✗'}\n"
            summary_text += f"  Image augmented: {'Yes' if meta['has_image_augmentation'] else 'No'}\n"
            summary_text += "\n"
        
        with open(epoch_dir / "epoch_summary.txt", 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        logger.info(f"Saved {len(metadata)} samples for epoch {epoch} to {epoch_dir}")
    
    def _create_comparison_image(
        self,
        original_img: Image.Image,
        augmented_img: Image.Image,
        save_path: Path
    ):
        """Create side-by-side comparison of original and augmented images."""
        # Resize images to same height if needed
        max_height = max(original_img.height, augmented_img.height)
        
        if original_img.height != max_height:
            ratio = max_height / original_img.height
            original_img = original_img.resize(
                (int(original_img.width * ratio), max_height),
                Image.Resampling.LANCZOS
            )
        
        if augmented_img.height != max_height:
            ratio = max_height / augmented_img.height
            augmented_img = augmented_img.resize(
                (int(augmented_img.width * ratio), max_height),
                Image.Resampling.LANCZOS
            )
        
        # Create comparison image
        total_width = original_img.width + augmented_img.width + 10  # 10px gap
        comparison = Image.new('RGB', (total_width, max_height + 40), color='white')
        
        # Paste images
        comparison.paste(original_img, (0, 40))
        comparison.paste(augmented_img, (original_img.width + 10, 40))
        
        # Add labels
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw labels
        draw.text((original_img.width // 2 - 40, 10), "Original", fill='black', font=font)
        draw.text((original_img.width + 10 + augmented_img.width // 2 - 50, 10), "Augmented", fill='black', font=font)
        
        comparison.save(save_path)
    
    def _create_text_summary(self, save_path: Path, metadata: Dict[str, Any]):
        """Create a text summary file for a single sample."""
        summary = "Sample Summary\n"
        summary += "=" * 50 + "\n\n"
        
        summary += f"Original Question:\n  {metadata['original_question']}\n\n"
        
        if metadata['has_text_augmentation']:
            summary += f"Augmented Question:\n  {metadata['augmented_question']}\n\n"
        
        summary += f"Ground Truth Answer:\n  {metadata['ground_truth']}\n\n"
        summary += f"Model Prediction:\n  {metadata['model_prediction']}\n\n"
        summary += f"Prediction Status: {'CORRECT ✓' if metadata['is_correct'] else 'INCORRECT ✗'}\n\n"
        
        summary += "Augmentation Applied:\n"
        summary += f"  - Image: {'Yes' if metadata['has_image_augmentation'] else 'No'}\n"
        summary += f"  - Text: {'Yes' if metadata['has_text_augmentation'] else 'No'}\n"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(summary)


def create_sample_observer(save_dir: str, num_samples: int = 5) -> SampleObserver:
    """
    Factory function to create a SampleObserver.
    
    Args:
        save_dir: Directory to save observations
        num_samples: Number of samples to save per epoch
        
    Returns:
        SampleObserver instance
    """
    return SampleObserver(save_dir, num_samples)


class WrongPredictionTracker:
    """
    Tracks all incorrectly predicted samples during validation/evaluation.
    
    This tracker helps monitor which samples the model struggles with
    and provides detailed information about prediction errors.
    """
    
    def __init__(self, save_dir: str):
        """
        Initialize the wrong prediction tracker.
        
        Args:
            save_dir: Directory to save wrong predictions
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"WrongPredictionTracker initialized: saving to {save_dir}")
    
    def save_wrong_predictions(
        self,
        epoch: int,
        wrong_samples: List[Dict[str, Any]],
        label_decoder: Dict[int, str]
    ):
        """
        Save all wrong predictions for an epoch.
        
        Args:
            epoch: Current epoch number
            wrong_samples: List of wrong prediction dictionaries containing:
                - idx: sample index in dataset
                - image_path: path to image file
                - question: question text
                - prediction: int (predicted label index)
                - ground_truth: int (true label index)
                - logits: prediction logits (optional)
            label_decoder: Dictionary mapping label indices to answer strings
        """
        epoch_dir = self.save_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(exist_ok=True)
        
        # Save metadata in JSON
        metadata = []
        
        for sample in wrong_samples:
            # Decode predictions and ground truth
            pred_answer = label_decoder.get(sample['prediction'], f"unknown_{sample['prediction']}")
            gt_answer = label_decoder.get(sample['ground_truth'], f"unknown_{sample['ground_truth']}")
            
            # Prepare metadata entry
            sample_meta = {
                'sample_idx': sample['idx'],
                'image_path': sample['image_path'],
                'question': sample['question'],
                'model_prediction': pred_answer,
                'ground_truth': gt_answer,
                'prediction_label': int(sample['prediction']),
                'ground_truth_label': int(sample['ground_truth']),
            }
            
            # Add logits if available for deeper analysis
            if 'logits' in sample and sample['logits'] is not None:
                # Convert logits to list for JSON serialization
                if isinstance(sample['logits'], torch.Tensor):
                    logits_list = sample['logits'].cpu().tolist()
                else:
                    logits_list = sample['logits']
                
                # Get top-5 predictions
                top5_indices = sorted(range(len(logits_list)), key=lambda i: logits_list[i], reverse=True)[:5]
                top5_predictions = [
                    {
                        'rank': i + 1,
                        'answer': label_decoder.get(idx, f"unknown_{idx}"),
                        'label': idx,
                        'score': float(logits_list[idx])
                    }
                    for i, idx in enumerate(top5_indices)
                ]
                sample_meta['top5_predictions'] = top5_predictions
            
            metadata.append(sample_meta)
        
        # Save epoch-level metadata
        epoch_metadata = {
            'epoch': epoch,
            'num_wrong_predictions': len(wrong_samples),
            'wrong_samples': metadata
        }
        
        with open(epoch_dir / "wrong_predictions.json", 'w', encoding='utf-8') as f:
            json.dump(epoch_metadata, f, ensure_ascii=False, indent=2)
        
        # Create human-readable summary
        summary_text = f"Epoch {epoch} - Wrong Predictions Summary\n"
        summary_text += "=" * 80 + "\n\n"
        summary_text += f"Total wrong predictions: {len(metadata)}\n\n"
        summary_text += "-" * 80 + "\n"
        
        for i, meta in enumerate(metadata, 1):
            summary_text += f"\n[{i}] Sample Index: {meta['sample_idx']}\n"
            summary_text += f"Image: {meta['image_path']}\n"
            summary_text += f"Question: {meta['question']}\n"
            summary_text += f"Ground Truth: {meta['ground_truth']}\n"
            summary_text += f"Prediction: {meta['model_prediction']}\n"
            
            if 'top5_predictions' in meta:
                summary_text += f"Top-5 Predictions:\n"
                for pred in meta['top5_predictions']:
                    summary_text += f"  {pred['rank']}. {pred['answer']} (score: {pred['score']:.4f})\n"
            
            summary_text += "-" * 80 + "\n"
        
        with open(epoch_dir / "wrong_predictions_summary.txt", 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        logger.info(f"Saved {len(metadata)} wrong predictions for epoch {epoch} to {epoch_dir}")
        
        return len(metadata)


def create_wrong_prediction_tracker(save_dir: str) -> WrongPredictionTracker:
    """
    Factory function to create a WrongPredictionTracker.
    
    Args:
        save_dir: Directory to save wrong predictions
        
    Returns:
        WrongPredictionTracker instance
    """
    return WrongPredictionTracker(save_dir)
