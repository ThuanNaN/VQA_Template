import numpy as np
from scipy.special import betainc
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def lambda_from_rank(rank, batch_size, s=10.0, a=0.5):
    """
    Calculate lambda value from rank using beta incomplete function.
    
    Args:
        rank: Rank of the sample (1-indexed)
        batch_size: Total number of samples in batch
        s: Shape parameter for beta distribution
        a: Asymmetry parameter for beta distribution
        
    Returns:
        Lambda value for sample weighting
    """
    x = rank / float(batch_size)
    alpha = s * (1.0 - a)
    beta = s * a
    alpha = max(alpha, 1e-6)
    beta = max(beta, 1e-6)
    lam = 1.0 - betainc(alpha, beta, x)
    return lam

def compute_lambdas(sample_losses, s=10.0, a=0.5):
    """
    Compute lambda values for all samples based on their losses.
    
    Args:
        sample_losses: Dictionary mapping sample IDs to their loss values
        s: Shape parameter for beta distribution
        a: Asymmetry parameter for beta distribution
        
    Returns:
        Dictionary mapping sample IDs to their lambda values
    """
    sorted_ids = sorted(sample_losses.keys(), key=lambda k: sample_losses[k])
    B = len(sorted_ids)
    lambdas = {}
    for rank, sid in enumerate(sorted_ids, start=1):
        lam = lambda_from_rank(rank, B, s, a)
        lambdas[sid] = lam
    return lambdas
