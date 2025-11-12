"""
Base classes for image and text augmentation.

This module provides abstract base classes for implementing augmentation
strategies with curriculum learning support.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
from PIL import Image
from enum import Enum


class DifficultyLevel(Enum):
    """
    Difficulty levels for curriculum learning.
    
    EASY: Minimal augmentation - slight changes
    MEDIUM: Moderate augmentation - partial transformations
    HARD: Aggressive augmentation - heavy transformations
    """
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class BaseAugmentation(ABC):
    """
    Abstract base class for all augmentation strategies.
    
    This class defines the interface that all augmentation implementations
    must follow, enabling flexible augmentation strategies with curriculum
    learning support.
    
    Args:
        difficulty: Difficulty level for augmentation
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        difficulty: Union[DifficultyLevel, str] = DifficultyLevel.EASY,
        seed: Optional[int] = None
    ):
        """Initialize the augmentation strategy."""
        if isinstance(difficulty, str):
            difficulty = DifficultyLevel(difficulty.lower())
        
        self.difficulty = difficulty
        self.seed = seed
        self._configure_parameters()
    
    @abstractmethod
    def _configure_parameters(self):
        """
        Configure augmentation parameters based on difficulty level.
        
        This method should set instance variables that control the strength
        of augmentation operations.
        """
        pass
    
    @abstractmethod
    def augment(self, data: Any, **kwargs) -> Any:
        """
        Apply augmentation to the input data.
        
        Args:
            data: Input data to augment (e.g., PIL Image, text string)
            **kwargs: Additional augmentation options
            
        Returns:
            Augmented data
        """
        pass
    
    @abstractmethod
    def get_augmentation_info(self) -> dict:
        """
        Get information about current augmentation configuration.
        
        Returns:
            Dictionary with augmentation parameters
        """
        pass
    
    def set_difficulty(self, difficulty: Union[DifficultyLevel, str]):
        """
        Update the difficulty level and reconfigure parameters.
        
        Args:
            difficulty: New difficulty level
        """
        if isinstance(difficulty, str):
            difficulty = DifficultyLevel(difficulty.lower())
        
        self.difficulty = difficulty
        self._configure_parameters()


class BaseImageAugmentation(BaseAugmentation):
    """
    Abstract base class for image augmentation strategies.
    
    Extends BaseAugmentation with image-specific functionality.
    """
    
    @abstractmethod
    def augment(self, image: Image.Image, **kwargs) -> Image.Image:
        """
        Apply augmentation to an image.
        
        Args:
            image: PIL Image to augment
            **kwargs: Additional augmentation options
            
        Returns:
            Augmented PIL Image
        """
        pass


class BaseTextAugmentation(BaseAugmentation):
    """
    Abstract base class for text augmentation strategies.
    
    Extends BaseAugmentation with text-specific functionality.
    """
    
    @abstractmethod
    def augment(self, text: str, **kwargs) -> str:
        """
        Apply augmentation to text.
        
        Args:
            text: Text string to augment
            **kwargs: Additional augmentation options
            
        Returns:
            Augmented text string
        """
        pass


class NoAugmentation(BaseAugmentation):
    """
    No-op augmentation that returns data unchanged.
    
    Useful for disabling augmentation or as a default strategy.
    """
    
    def _configure_parameters(self):
        """No parameters needed for no-op."""
        pass
    
    def augment(self, data: Any, **kwargs) -> Any:
        """Return data unchanged."""
        return data
    
    def get_augmentation_info(self) -> dict:
        """Return info indicating no augmentation."""
        return {
            "type": "NoAugmentation",
            "difficulty": self.difficulty.value,
        }
