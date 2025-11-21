"""
Base classes for image and text augmentation.

This module provides abstract base classes for implementing augmentation
strategies with curriculum learning support.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union, List
from PIL import Image


class BaseAugmentation(ABC):
    """
    Abstract base class for all augmentation strategies.
    
    This class defines the interface that all augmentation implementations
    must follow, enabling flexible augmentation strategies with curriculum
    learning support.
    
    Args:
        difficulty: Difficulty level for augmentation (float 0.0-1.0)
                   0.0 = easiest (minimal augmentation)
                   1.0 = hardest (maximum augmentation)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        difficulty: Union[int, float] = 0.0,
        seed: Optional[int] = None
    ):
        """Initialize the augmentation strategy."""
        # Ensure difficulty is float in valid range
        self.difficulty = float(max(0.0, min(1.0, difficulty)))
        self.seed = seed
        self._configure_parameters()
    
    @abstractmethod
    def _configure_parameters(self):
        """
        Configure augmentation parameters based on difficulty level.
        
        This method should set instance variables that control the strength
        of augmentation operations. Use self.difficulty (0.0-1.0) to interpolate
        parameters smoothly.
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
    
    def set_difficulty(self, difficulty: Union[int, float]):
        """
        Update the difficulty level and reconfigure parameters.
        
        Args:
            difficulty: New difficulty level (float 0.0-1.0)
        """
        self.difficulty = float(max(0.0, min(1.0, difficulty)))
        self._configure_parameters()


class BaseImageAugmentation(BaseAugmentation):
    """
    Abstract base class for image augmentation strategies.
    
    Extends BaseAugmentation with image-specific functionality.
    All image augmentations must return a list of PIL Images for multi-view support.
    """
    
    @abstractmethod
    def augment(self, image: Image.Image, **kwargs) -> List[Image.Image]:
        """
        Apply augmentation to an image.
        
        Args:
            image: PIL Image to augment
            **kwargs: Additional augmentation options
            
        Returns:
            List of augmented PIL Images (multi-view).
            If no augmentation, returns list with single original image.
        """
        pass


class BaseTextAugmentation(BaseAugmentation):
    """
    Abstract base class for text augmentation strategies.
    
    Extends BaseAugmentation with text-specific functionality.
    All text augmentations must return a list of strings for multi-view support.
    """
    
    @abstractmethod
    def augment(self, text: str, **kwargs) -> List[str]:
        """
        Apply augmentation to text.
        
        Args:
            text: Text string to augment
            **kwargs: Additional augmentation options
            
        Returns:
            List of augmented text strings (multi-view).
            If no augmentation, returns list with single original text.
        """
        pass


class NoAugmentation(BaseAugmentation):
    """
    No-op augmentation that returns data unchanged.
    
    Useful for disabling augmentation or as a default strategy.
    Returns a list containing only the original data for consistency.
    """
    
    def _configure_parameters(self):
        """No parameters needed for no-op."""
        pass
    
    def augment(self, data: Any, **kwargs) -> List[Any]:
        """Return list containing original data unchanged."""
        return [data]
    
    def get_augmentation_info(self) -> dict:
        """Return info indicating no augmentation."""
        return {
            "type": "NoAugmentation",
            "difficulty": self.difficulty,
        }
