"""
Script chọn ra các câu hỏi khó (high entropy hoặc low confidence).
"""
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def select_hard_samples(
    samples: List[Dict],
    top_percent: float = 0.1,
    method: str = 'entropy'
) -> List[Dict]:
    """
    Chọn ra các mẫu khó dựa trên entropy hoặc confidence.
    
    Args:
        samples: Danh sách các mẫu đã tính difficulty
        top_percent: Tỷ lệ phần trăm mẫu khó cần lấy (0.0-1.0)
        method: Phương pháp chọn ('entropy' hoặc 'confidence')
        
    Returns:
        Danh sách các mẫu khó được chọn
    """
    if method == 'entropy':
        # Sắp xếp theo entropy giảm dần (cao = khó)
        sorted_samples = sorted(samples, key=lambda x: x['entropy'], reverse=True)
        metric_name = 'entropy'
    elif method == 'confidence':
        # Sắp xếp theo confidence tăng dần (thấp = khó)
        sorted_samples = sorted(samples, key=lambda x: x['confidence'])
        metric_name = 'confidence'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Lấy top percent
    num_hard = int(len(sorted_samples) * top_percent)
    hard_samples = sorted_samples[:num_hard]
    
    logger.info(f"Chọn {num_hard}/{len(samples)} mẫu khó (top {top_percent*100}% {metric_name})")
    
    # Log statistics
    if method == 'entropy':
        values = [s['entropy'] for s in hard_samples]
        logger.info(f"  Entropy range: [{min(values):.3f}, {max(values):.3f}]")
    else:
        values = [s['confidence'] for s in hard_samples]
        logger.info(f"  Confidence range: [{min(values):.3f}, {max(values):.3f}]")
    
    return hard_samples
