"""Scheduler for Curriculum Learning in Text and Image Augmentation."""

import math
from typing import Literal


class CurriculumScheduler:
    """
    Smooth curriculum learning scheduler with continuous difficulty adjustment.
    
    This scheduler provides smooth, continuous difficulty progression similar to
    learning rate schedulers, allowing for more fine-grained control over the
    curriculum learning process. Returns a float value between 0.0 (easiest) and
    1.0 (hardest) that can be used to interpolate augmentation parameters.
    
    Supported scheduling strategies:
    - linear: Linear progression from 0 to 1
    - cosine: Cosine annealing-style smooth progression
    - exponential: Exponential growth in difficulty
    - step: Step-wise increases at specified intervals
    - polynomial: Polynomial progression with configurable power
    
    Args:
        total_epochs: Total number of training epochs
        strategy: Scheduling strategy ('linear', 'cosine', 'exponential', 'step', 'polynomial')
        min_difficulty: Minimum difficulty value (default: 0.0)
        max_difficulty: Maximum difficulty value (default: 1.0)
        warmup_epochs: Number of epochs to keep at minimum difficulty (default: 0)
        **kwargs: Strategy-specific parameters:
            - exponential: gamma (default: 0.1)
            - step: step_size (default: total_epochs // 3), gamma (default: 0.33)
            - polynomial: power (default: 2.0)
            
    Example:
        >>> # Linear progression
        >>> scheduler = CurriculumScheduler(total_epochs=100, strategy='linear')
        >>> 
        >>> # Cosine annealing
        >>> scheduler = CurriculumScheduler(total_epochs=100, strategy='cosine')
        >>> 
        >>> # Exponential with custom gamma
        >>> scheduler = CurriculumScheduler(
        ...     total_epochs=100, strategy='exponential', gamma=0.05
        ... )
        >>> 
        >>> # Use in training loop
        >>> for epoch in range(100):
        ...     difficulty = scheduler.get_difficulty(epoch)
        ...     # Use difficulty to interpolate augmentation parameters
        ...     # e.g., mask_ratio = 0.1 + difficulty * 0.4  # 0.1 to 0.5
    """
    
    def __init__(
        self,
        total_epochs: int,
        strategy: Literal['linear', 'cosine', 'exponential', 'step', 'polynomial'] = 'linear',
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0,
        warmup_epochs: int = 0,
        **kwargs
    ):
        """Initialize the smooth curriculum scheduler."""
        self.total_epochs = total_epochs
        self.strategy = strategy
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.warmup_epochs = warmup_epochs
        
        # Strategy-specific parameters
        self.gamma = kwargs.get('gamma', 0.1)  # For exponential and step
        self.step_size = kwargs.get('step_size', total_epochs // 3)  # For step
        self.power = kwargs.get('power', 2.0)  # For polynomial
        
        # Validate
        if not 0.0 <= min_difficulty <= max_difficulty <= 1.0:
            raise ValueError("Difficulty values must satisfy: 0 ≤ min_difficulty ≤ max_difficulty ≤ 1")
        if warmup_epochs >= total_epochs:
            raise ValueError(f"Warmup epochs ({warmup_epochs}) must be less than total epochs ({total_epochs})")
    
    def get_difficulty(self, epoch: int) -> float:
        """
        Get the continuous difficulty value for a given epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            Float value between min_difficulty and max_difficulty representing
            the augmentation difficulty for this epoch
        """
        # Warmup phase
        if epoch < self.warmup_epochs:
            return self.min_difficulty
        
        # Adjust epoch for warmup
        adjusted_epoch = epoch - self.warmup_epochs
        adjusted_total = self.total_epochs - self.warmup_epochs
        
        # Calculate progress (0 to 1)
        if self.strategy == 'linear':
            progress = adjusted_epoch / max(adjusted_total - 1, 1)
            
        elif self.strategy == 'cosine':
            # Cosine annealing: smooth S-curve from 0 to 1
            progress = (1 - math.cos(math.pi * adjusted_epoch / max(adjusted_total - 1, 1))) / 2
            
        elif self.strategy == 'exponential':
            # Exponential growth: difficulty = gamma^((total-epoch)/total)
            # Inverted so difficulty increases over time
            progress = 1 - self.gamma ** ((adjusted_total - adjusted_epoch) / adjusted_total)
            
        elif self.strategy == 'step':
            # Step-wise increases
            num_steps = adjusted_epoch // self.step_size
            progress = min(num_steps * self.gamma, 1.0)
            
        elif self.strategy == 'polynomial':
            # Polynomial progression: (epoch/total)^power
            progress = (adjusted_epoch / max(adjusted_total - 1, 1)) ** self.power
            
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Scale to min/max difficulty
        return self.min_difficulty + progress * (self.max_difficulty - self.min_difficulty)
    
    def get_schedule_info(self) -> dict:
        """
        Get information about the curriculum schedule.
        
        Returns:
            Dictionary with schedule information and sample difficulty values
        """
        # Sample difficulties at key epochs
        sample_epochs = [0, self.total_epochs // 4, self.total_epochs // 2, 
                        3 * self.total_epochs // 4, self.total_epochs - 1]
        samples = {f"epoch_{ep}": round(self.get_difficulty(ep), 4) for ep in sample_epochs}
        
        return {
            "strategy": self.strategy,
            "total_epochs": self.total_epochs,
            "min_difficulty": self.min_difficulty,
            "max_difficulty": self.max_difficulty,
            "warmup_epochs": self.warmup_epochs,
            "samples": samples,
            "parameters": {
                "gamma": self.gamma if self.strategy in ['exponential', 'step'] else None,
                "step_size": self.step_size if self.strategy == 'step' else None,
                "power": self.power if self.strategy == 'polynomial' else None,
            }
        }
