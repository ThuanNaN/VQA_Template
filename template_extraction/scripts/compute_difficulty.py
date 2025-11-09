"""
Script tính độ khó (entropy, confidence) cho từng câu hỏi trong dataset VQA.
"""
import torch
import numpy as np
from typing import Dict, List
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_entropy(probs: np.ndarray) -> float:
    """
    Tính entropy của phân phối xác suất.
    
    Args:
        probs: Mảng xác suất (đã softmax)
        
    Returns:
        Giá trị entropy
    """
    # Tránh log(0)
    probs = probs + 1e-10
    return -np.sum(probs * np.log(probs))


def compute_difficulty_scores(
    model,
    dataset,
    batch_size: int = 32,
    device: str = 'cuda'
) -> List[Dict]:
    """
    Tính độ khó cho từng mẫu trong dataset.
    
    Args:
        model: Model VQA đã huấn luyện
        dataset: Dataset cần đánh giá
        batch_size: Kích thước batch
        device: Thiết bị (cuda/cpu)
        
    Returns:
        Danh sách dict chứa question_id, question_text, gt_answer, entropy, confidence
    """
    model.eval()
    model.to(device)
    
    results = []
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Tính độ khó cho {len(dataset)} mẫu...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing difficulty")):
            # Move to device
            image = batch['image'].to(device)
            question_input_ids = batch['question_input_ids'].to(device)
            question_attention_mask = batch['question_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(
                image=image,
                question_input_ids=question_input_ids,
                question_attention_mask=question_attention_mask
            )
            
            # Get logits and compute softmax probabilities
            logits = outputs['logits']  # [batch_size, num_classes]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # Get predictions and ground truth
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            # Compute entropy and confidence for each sample
            for i in range(len(probs)):
                sample_probs = probs[i]
                entropy = compute_entropy(sample_probs)
                confidence = float(np.max(sample_probs))
                
                # Get question text and answer
                idx = batch_idx * batch_size + i
                question_text = dataset.data['questions'][idx]
                gt_answer = dataset.data['answers'][idx]
                img_path = dataset.data['img_paths'][idx]
                
                # Check if prediction is correct
                predicted_label = int(predictions[i])
                gt_label = int(labels[i])
                is_correct = (predicted_label == gt_label)
                
                # Get predicted answer text
                # Reverse lookup: label -> answer
                label_to_answer = {v: k for k, v in dataset.label_encoder.items()}
                predicted_answer = label_to_answer.get(predicted_label, "unknown")
                
                results.append({
                    'question_id': idx,
                    'question_text': question_text,
                    'gt_answer': gt_answer,
                    'predicted_answer': predicted_answer,
                    'image_path': img_path,
                    'entropy': float(entropy),
                    'confidence': confidence,
                    'predicted_label': predicted_label,
                    'gt_label': gt_label,
                    'is_correct': is_correct
                })
    
    logger.info(f"✅ Hoàn thành tính độ khó cho {len(results)} mẫu")
    
    # Log accuracy statistics
    correct_count = sum(1 for r in results if r['is_correct'])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    logger.info(f"📊 Model accuracy: {correct_count}/{total_count} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return results
