"""Scheduler for Curriculum Learning in Text and Image Augmentation."""

from typing import Optional
from .base import DifficultyLevel

class CurriculumLearningScheduler:
    """
    Scheduler for curriculum learning progression.
    
    Manages the transition from easy to hard samples during training,
    following a curriculum learning approach. This scheduler applies to
    both text and image augmentation strategies.
    
    Curriculum learning is a training strategy that mimics human learning
    by presenting training examples in a meaningful order - from easy to
    hard. This scheduler helps implement this strategy by determining the
    appropriate difficulty level for each training epoch.
    
    Args:
        total_epochs: Total number of training epochs
        easy_epochs: Number of epochs to train on easy samples
        medium_epochs: Number of epochs to train on medium samples
        hard_epochs: Number of epochs to train on hard samples (remaining epochs)
        
    Example:
        >>> scheduler = CurriculumLearningScheduler(total_epochs=30)
        >>> for epoch in range(30):
        ...     difficulty = scheduler.get_difficulty_for_epoch(epoch)
        ...     # Use difficulty to configure text or image augmentation
    """
    
    def __init__(
        self,
        total_epochs: int,
        easy_epochs: Optional[int] = None,
        medium_epochs: Optional[int] = None,
        hard_epochs: Optional[int] = None
    ):
        """Initialize the curriculum scheduler."""
        self.total_epochs = total_epochs
        
        # Default split: 30% easy, 30% medium, 40% hard
        if easy_epochs is None:
            easy_epochs = int(total_epochs * 0.3)
        if medium_epochs is None:
            medium_epochs = int(total_epochs * 0.3)
        if hard_epochs is None:
            hard_epochs = total_epochs - easy_epochs - medium_epochs
        
        self.easy_epochs = easy_epochs
        self.medium_epochs = medium_epochs
        self.hard_epochs = hard_epochs
        
        # Validate
        if easy_epochs + medium_epochs + hard_epochs != total_epochs:
            raise ValueError(
                f"Sum of easy ({easy_epochs}), medium ({medium_epochs}), "
                f"and hard ({hard_epochs}) epochs must equal total_epochs ({total_epochs})"
            )
    
    def get_difficulty_for_epoch(self, epoch: int) -> DifficultyLevel:
        """
        Get the difficulty level for a given epoch.
        
        This method determines which difficulty level should be used for
        augmentation (both text and image) at a specific epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            DifficultyLevel for the current epoch (EASY, MEDIUM, or HARD)
        """
        if epoch < self.easy_epochs:
            return DifficultyLevel.EASY
        elif epoch < self.easy_epochs + self.medium_epochs:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.HARD
    
    def get_schedule_info(self) -> dict:
        """
        Get information about the curriculum schedule.
        
        Returns:
            Dictionary with schedule information including epoch ranges
            for each difficulty level
        """
        return {
            "total_epochs": self.total_epochs,
            "easy_epochs": self.easy_epochs,
            "medium_epochs": self.medium_epochs,
            "hard_epochs": self.hard_epochs,
            "schedule": [
                f"Epochs 0-{self.easy_epochs-1}: EASY",
                f"Epochs {self.easy_epochs}-{self.easy_epochs+self.medium_epochs-1}: MEDIUM",
                f"Epochs {self.easy_epochs+self.medium_epochs}-{self.total_epochs-1}: HARD",
            ]
        }
