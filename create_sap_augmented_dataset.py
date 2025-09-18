import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor
import logging
from pathlib import Path
from collections import Counter
import time

from dataset import ViVQADataset, OpenViVQADataset, ViVQAXDataset
from models import SimpleVQAConfig, SimpleVQA
from augmentation.augment_client import VlmAugmentClient
from utils.sap_augment import compute_lambdas
import random
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_sample_losses(model, dataloader, device):
    """
    Compute individual sample losses for the entire dataset.
    """
    model.eval()
    sample_losses = {}
    sample_idx = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing sample losses"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Extract labels before removing from batch
            labels = batch.get("label")  # Get the labels first
            
            # Remove "label" from batch and add as "labels" for model compatibility
            if "label" in batch:
                batch["labels"] = batch.pop("label")  # Move label to labels and remove original
            
            # Forward pass
            outputs = model(**batch)
            
            # Compute individual losses
            logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
            
            if labels is not None and logits is not None:
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                individual_losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Store losses
                for loss_val in individual_losses.cpu().numpy():
                    sample_losses[sample_idx] = float(loss_val)
                    sample_idx += 1
    
    return sample_losses

def select_samples_for_augmentation(sample_losses, augment_ratio=0.3, s=10.0, a=0.5, max_augment_samples=None, 
                                   use_adaptive_augment=False, loss_threshold=None):
    """
    Select samples for augmentation based on SapAugment logic.
    
    Args:
        sample_losses: Dictionary of sample losses
        augment_ratio: Ratio of samples to augment
        s, a: SapAugment parameters
        max_augment_samples: Maximum number of samples to augment (overrides ratio if specified)
        use_adaptive_augment: Whether to use adaptive augmentation based on loss scores
        loss_threshold: Loss threshold for adaptive augmentation (samples above this get more augmentation)
        
    Returns:
        Set of sample indices to augment, lambda values, and augmentation strategy info
    """
    # Compute lambda values
    lambdas = compute_lambdas(sample_losses, s, a)
    
    # Sort by lambda values (descending - hardest samples first)
    sorted_samples = sorted(lambdas.items(), key=lambda x: x[1], reverse=True)
    
    if use_adaptive_augment:
        return _adaptive_sample_selection(sample_losses, sorted_samples, lambdas, 
                                        augment_ratio, max_augment_samples, loss_threshold)
    else:
        return _standard_sample_selection(sorted_samples, lambdas, augment_ratio, max_augment_samples)

def _standard_sample_selection(sorted_samples, lambdas, augment_ratio, max_augment_samples):
    """Standard sample selection based on ranking only."""
    # Determine number of samples to augment
    if max_augment_samples is not None:
        # Use fixed number if specified
        num_to_augment = min(max_augment_samples, len(sorted_samples))
        logger.info(f"Using fixed number of samples: {num_to_augment} (max: {max_augment_samples})")
    else:
        # Use ratio
        num_to_augment = int(len(sorted_samples) * augment_ratio)
        logger.info(f"Using ratio-based selection: {num_to_augment} samples ({augment_ratio:.1%} of {len(sorted_samples)})")
    
    # Select top samples for augmentation
    samples_to_augment = set([idx for idx, _ in sorted_samples[:num_to_augment]])
    
    logger.info(f"Selected {len(samples_to_augment)} samples for augmentation")
    if num_to_augment > 0:
        logger.info(f"Lambda range for selected samples: {sorted_samples[0][1]:.4f} - {sorted_samples[num_to_augment-1][1]:.4f}")
    
    augment_strategy = {
        "type": "standard",
        "total_samples": len(samples_to_augment),
        "augment_counts": {idx: 1 for idx in samples_to_augment}  # Each sample gets 1 augmentation
    }
    
    return samples_to_augment, lambdas, augment_strategy

def _adaptive_sample_selection(
    sample_losses, sorted_samples, lambdas,
    augment_ratio, max_augment_samples,
    loss_threshold
):
    """Adaptive sample selection with balanced difficulty quotas."""

    # ----- 1. Calculate loss statistics -----
    loss_values = list(sample_losses.values())
    mean_loss = np.mean(loss_values)
    std_loss = np.std(loss_values)

    if loss_threshold is None:
        loss_threshold = mean_loss + 0.5 * std_loss

    logger.info(f"Loss statistics: mean={mean_loss:.4f}, std={std_loss:.4f}, threshold={loss_threshold:.4f}")

    # ----- 2. Classify samples -----
    very_hard_samples = []  # loss > mean + std
    hard_samples = []       # loss > threshold
    medium_samples = []     # loss > mean
    easy_samples = []       # loss <= mean

    for idx, loss in sample_losses.items():
        lam = lambdas[idx]
        if loss > mean_loss + std_loss:
            very_hard_samples.append((idx, loss, lam))
        elif loss > loss_threshold:
            hard_samples.append((idx, loss, lam))
        elif loss > mean_loss:
            medium_samples.append((idx, loss, lam))
        else:
            easy_samples.append((idx, loss, lam))

    logger.info(
        f"Sample distribution: very_hard={len(very_hard_samples)}, "
        f"hard={len(hard_samples)}, medium={len(medium_samples)}, easy={len(easy_samples)}"
    )

    # ----- 3. Calculate maximum samples to select -----
    if max_augment_samples is not None:
        max_api_calls = max_augment_samples 
    else:
        max_api_calls = int(len(sample_losses) * augment_ratio)

    # ----- 4. Distribute quotas among difficulty groups -----
    difficulty_groups = [
        ("very_hard", very_hard_samples, 0.30, 4),
        ("hard", hard_samples, 0.30, 3),
        ("medium", medium_samples, 0.25, 2),
        ("easy", easy_samples, 0.15, 1),
    ]

    samples_to_augment = set()
    augment_counts = {}

    for group, quota, aug_per_sample in difficulty_groups:
        if not group:
            continue
        num_to_pick = int(max_api_calls * quota)
        num_to_pick = min(num_to_pick, len(group))
        if num_to_pick <= 0:
            continue

        weights = np.array([g[2] for g in group], dtype=float)
        weights /= weights.sum()
        chosen = np.random.choice([g[0] for g in group], size=num_to_pick, replace=False, p=weights)

        for idx in chosen:
            idx = int(idx)
            samples_to_augment.add(idx)
            augment_counts[idx] = aug_per_sample

    remaining_budget = max_api_calls - len(samples_to_augment)
    if remaining_budget > 0:
        remaining = [x for x in sample_losses.keys() if x not in samples_to_augment]
        if remaining:
            extra_chosen = random.sample(remaining, min(remaining_budget, len(remaining)))
            for idx in extra_chosen:
                idx = int(idx)
                samples_to_augment.add(idx)
                augment_counts[idx] = 1  # dễ nhất cho phần dư

    # ----- 5. Metadata for selected samples -----
    total_expected_augmentations = sum(augment_counts.values())
    difficulty_distribution = {
        "very_hard": len([x for x,_,_ in very_hard_samples if x in samples_to_augment]),
        "hard": len([x for x,_,_ in hard_samples if x in samples_to_augment]),
        "medium": len([x for x,_,_ in medium_samples if x in samples_to_augment]),
        "easy": len([x for x,_,_ in easy_samples if x in samples_to_augment])
    }

    logger.info(f"Adaptive augmentation: {len(samples_to_augment)} samples selected ({len(samples_to_augment)} API calls)")
    logger.info(f"Expected total augmentations: {total_expected_augmentations}")
    logger.info(f"Augmentation distribution: {dict(sorted(Counter(augment_counts.values()).items()))}")

    augment_strategy = {
        "type": "adaptive_quota",
        "total_samples": len(samples_to_augment),
        "api_calls_needed": len(samples_to_augment),
        "total_augmentations": total_expected_augmentations,
        "augment_counts": augment_counts,
        "loss_threshold": loss_threshold,
        "difficulty_distribution": difficulty_distribution
    }

    return samples_to_augment, lambdas, augment_strategy

def generate_augmented_dataset(dataset, samples_to_augment, gemini_client, output_path, augment_strategy, lambdas, request_delay=2.0):
    """
    Generate augmented dataset and save to file.
    
    Args:
        request_delay: Delay in seconds between Gemini API requests to avoid rate limits (default: 2.0)
        budget: Maximum number of Gemini API calls (not total samples)
    """
    augmented_data = []
    failed_augmentations = 0
    total_expected = sum(augment_strategy["augment_counts"].values())
    api_calls_count = 0
    logger.info(f"Generating {total_expected} augmentations for {len(samples_to_augment)} samples")
    logger.info(f"Using {request_delay}s delay between requests to avoid rate limits")
    for idx in tqdm(samples_to_augment, desc="Generating augmented samples"):
        try:
            # Get original data
            original_data = dataset.data
            img_path = original_data['img_paths'][idx]
            original_question = original_data['questions'][idx]
            original_answer = original_data['answers'][idx]
            
            # Load image
            from PIL import Image
            pil_image = Image.open(img_path).convert('RGB')
            
            # Get lambda value for this sample (controls augmentation intensity)
            sample_lambda = lambdas.get(idx, 0.5)  # Default lambda if not found
            
            # Get number of augmentations for this sample
            num_augmentations = augment_strategy["augment_counts"].get(idx, 1)
            
            logger.debug(f"Sample {idx}: lambda={sample_lambda:.3f}, augmentations={num_augmentations}")
            
            try:
                # Make ONE API call to get ALL paraphrases for this sample
                # Higher lambda = more diverse/stronger paraphrases
                augmented_questions_text = gemini_client.generate_questions(
                    pil_image, original_question, lamda=sample_lambda
                )
                api_calls_count += 1  # Increment API call counter
                
                # Parse all questions from the text response
                augmented_questions_list = parse_augmented_question(augmented_questions_text, aug_idx=None)
                
                logger.debug(f"API call {api_calls_count}: Generated {len(augmented_questions_list)} paraphrases")
                
                # Use all generated paraphrases (up to num_augmentations)
                paraphrases_to_use = augmented_questions_list[:num_augmentations]
                
                # Create augmented entries for each paraphrase
                for aug_idx, augmented_question in enumerate(paraphrases_to_use):
                    augmented_entry = {
                        "original_idx": idx,
                        "augmentation_idx": aug_idx,
                        "img_path": img_path,
                        "original_question": original_question,
                        "augmented_question": augmented_question,
                        "answer": original_answer,
                        "lambda_value": sample_lambda,
                        "augmentation_source": "sap_augment_gemini",
                        "difficulty_score": augment_strategy.get("difficulty_scores", {}).get(idx, "unknown")
                    }
                    
                    augmented_data.append(augmented_entry)
                
                # Add delay between requests to avoid rate limits
                if request_delay > 0:
                    time.sleep(request_delay)
                
            except Exception as e:
                logger.warning(f"Failed to generate augmentations for sample {idx}: {e}")
                failed_augmentations += num_augmentations
                # Still sleep on error to avoid hitting rate limits
                if request_delay > 0:
                    time.sleep(request_delay)
                continue
            
        except Exception as e:
            logger.warning(f"Failed to process sample {idx}: {e}")
            failed_augmentations += num_augmentations
            continue
    
    # Save augmented dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(augmented_data)} augmented samples to {output_path}")
    logger.info(f"Used {api_calls_count} Gemini API calls")
    logger.info(f"Failed augmentations: {failed_augmentations}")
    if total_expected > 0:
        logger.info(f"Success rate: {(len(augmented_data)/(total_expected))*100:.1f}%")
    
    return augmented_data

def parse_augmented_question(augmented_text, aug_idx=None):
    """
    Parse augmented questions from Gemini response.
    Gemini returns numbered list based on lambda value.
    
    Args:
        augmented_text: Text response from Gemini
        aug_idx: If specified, return only the question at this index. If None, return all questions.
    
    Returns:
        If aug_idx is None: List of all parsed questions
        If aug_idx is specified: Single question at that index
    """
    lines = augmented_text.strip().split('\n')
    questions = []
    
    # Extract numbered questions
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            # Remove numbering (1., 2., etc.)
            if '.' in line:
                question = line.split('.', 1)[-1].strip()
            else:
                question = line[1:].strip()  # Remove first character if it's a number
            if question:
                questions.append(question)
    
    # If no numbered questions found, try to extract from plain text
    if not questions:
        for line in lines:
            line = line.strip()
            if line and not line.startswith('λ') and not line.startswith('Sentence:'):
                questions.append(line)
    
    # Return based on aug_idx parameter
    if aug_idx is None:
        return questions  # Return all questions
    else:
        # Return the requested question index, or first one if index out of range
        if questions:
            return questions[min(aug_idx, len(questions) - 1)]
        return "What is in the image?"  # Ultimate fallback

def main():
    parser = argparse.ArgumentParser(description="Generate offline augmented dataset using SapAugment")
    parser.add_argument('--dataset_name', type=str, choices=['ViVQA', 'OpenViVQA', 'ViVQA-X'],
                        default='ViVQA-X', help='Dataset name')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model for computing losses')
    parser.add_argument('--vis_model_name', type=str, default='google/vit-base-patch16-224',
                        help='Vision model name')
    parser.add_argument('--text_model_name', type=str, default='vinai/bartpho-syllable-base',
                        help='Text model name')
    parser.add_argument('--augment_ratio', type=float, default=0.3,
                        help='Ratio of samples to augment (default: 0.3)')
    parser.add_argument('--max_augment_samples', type=int, default=None,
                        help='Maximum number of samples to augment (overrides ratio if specified)')
    parser.add_argument('--sap_s', type=float, default=10.0,
                        help='SapAugment s parameter')
    parser.add_argument('--sap_a', type=float, default=0.5,
                        help='SapAugment a parameter')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for loss computation')
    parser.add_argument('--output_dir', type=str, default='augmented_datasets',
                        help='Output directory for augmented datasets')
    parser.add_argument('--use_adaptive_augment', action='store_true',
                        help='Use adaptive augmentation based on loss scores (hard samples get more augmentations)')
    parser.add_argument('--loss_threshold', type=float, default=None,
                        help='Loss threshold for adaptive augmentation (default: mean + 0.5*std)')
    parser.add_argument('--request_delay', type=float, default=20.0,
                        help='Delay in seconds between Gemini API requests to avoid rate limits (default: 2.0)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load processors
    vis_processor = AutoProcessor.from_pretrained(args.vis_model_name, use_fast=True)
    text_processor = AutoTokenizer.from_pretrained(args.text_model_name)
    
    # Load dataset
    if args.dataset_name == 'ViVQA':
        dataset = ViVQADataset(
            ann_path="data/vivqa/train.csv",
            img_dir="data/vivqa/images",
            text_processor=text_processor,
            vis_processor=vis_processor,
            max_length=64
        )
    elif args.dataset_name == 'OpenViVQA':
        dataset = OpenViVQADataset(
            ann_path="data/openvivqa/vlsp2023_train_data.json",
            img_dir="data/openvivqa/training-images",
            text_processor=text_processor,
            vis_processor=vis_processor,
            max_length=64
        )
    elif args.dataset_name == 'ViVQA-X':
        dataset = ViVQAXDataset(
            ann_path="data/vivqa-x/ViVQA-X_train.json",
            img_dir="data/MSCOCO/train2014",
            text_processor=text_processor,
            vis_processor=vis_processor,
            max_length=64
        )
    
    # Load model
    config = SimpleVQAConfig(
        vis_model_name=args.vis_model_name,
        text_model_name=args.text_model_name,
        num_classes=len(dataset.label_encoder)
    )
    model = SimpleVQA(config)

    model_path = args.model_path
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "pytorch_model.bin")
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    logger.info(f"Loaded model from {args.model_path}")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Step 1: Compute sample losses
    logger.info("Step 1: Computing sample losses...")
    sample_losses = compute_sample_losses(model, dataloader, device)
    
    # Step 2: Select samples for augmentation
    logger.info("Step 2: Selecting samples for augmentation...")
    samples_to_augment, lambdas, augment_strategy = select_samples_for_augmentation(
        sample_losses, args.augment_ratio, args.sap_s, args.sap_a, args.max_augment_samples,
        args.use_adaptive_augment, args.loss_threshold
    )
    
    # Step 3: Initialize Gemini client
    logger.info("Step 3: Initializing Gemini client...")
    import dotenv
    dotenv.load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    gemini_client = VlmAugmentClient(api_key=api_key)
    
    # Step 4: Generate augmented dataset
    logger.info("Step 4: Generating augmented dataset...")
    output_dir = Path(args.output_dir) / args.dataset_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    augmented_output_path = output_dir / f"{args.dataset_name}_sap_augmented.json"
    augmented_data = generate_augmented_dataset(
        dataset, samples_to_augment, gemini_client, augmented_output_path, augment_strategy, lambdas, args.request_delay
    )
    
    # Save metadata
    metadata = {
        "dataset_name": args.dataset_name,
        "model_path": args.model_path,
        "augment_ratio": args.augment_ratio,
        "max_augment_samples": args.max_augment_samples,
        "use_adaptive_augment": args.use_adaptive_augment,
        "loss_threshold": args.loss_threshold,
        "sap_s": args.sap_s,
        "sap_a": args.sap_a,
        "total_samples": len(dataset),
        "augmented_samples": len(samples_to_augment),
        "total_augmentations": len(augmented_data),
        "actual_augment_ratio": len(samples_to_augment) / len(dataset),
        "augmentation_strategy": augment_strategy,
        "samples_to_augment": list(samples_to_augment),
        "lambda_stats": {
            "min": min(lambdas.values()),
            "max": max(lambdas.values()),
            "mean": np.mean(list(lambdas.values()))
        }
    }
    
    metadata_path = output_dir / f"{args.dataset_name}_sap_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_path}")
    logger.info("Offline SapAugment dataset creation completed!")

if __name__ == "__main__":
    main()
