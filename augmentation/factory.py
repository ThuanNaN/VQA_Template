"""
Factory for creating augmentation strategies.

This module provides a factory pattern implementation for creating
augmentation objects based on type and configuration.
"""

from typing import Optional, Union
from .base import BaseImageAugmentation, BaseTextAugmentation, DifficultyLevel, NoAugmentation
from .visual.mask import MaskedImageAugmentation
from .textual.rule_based import RuleBasedTextAugmentation


class AugmentationFactory:
    """
    Factory for creating augmentation strategies.
    
    This factory provides a centralized way to create augmentation objects
    with consistent configuration and easy extensibility.
    
    Example:
        >>> factory = AugmentationFactory()
        >>> augmentor = factory.create_image_augmentation(
        ...     augmentation_type='masked',
        ...     difficulty='medium',
        ...     patch_size=16
        ... )
    """
    
    # Registry of available augmentation types
    _IMAGE_AUGMENTATIONS = {
        'masked': MaskedImageAugmentation,
        'none': NoAugmentation,
    }
    
    _TEXT_AUGMENTATIONS = {
        'rule-based': RuleBasedTextAugmentation,
        'none': NoAugmentation,
    }
    
    @classmethod
    def register_image_augmentation(cls, name: str, augmentation_class: type):
        """
        Register a new image augmentation type.
        
        Args:
            name: Name identifier for the augmentation
            augmentation_class: Class implementing BaseImageAugmentation
        """
        if not issubclass(augmentation_class, BaseImageAugmentation):
            raise ValueError(
                f"{augmentation_class} must inherit from BaseImageAugmentation"
            )
        cls._IMAGE_AUGMENTATIONS[name.lower()] = augmentation_class
    
    @classmethod
    def register_text_augmentation(cls, name: str, augmentation_class: type):
        """
        Register a new text augmentation type.
        
        Args:
            name: Name identifier for the augmentation
            augmentation_class: Class implementing BaseTextAugmentation
        """
        if not issubclass(augmentation_class, BaseTextAugmentation):
            raise ValueError(
                f"{augmentation_class} must inherit from BaseTextAugmentation"
            )
        cls._TEXT_AUGMENTATIONS[name.lower()] = augmentation_class
    
    @classmethod
    def create_image_augmentation(
        cls,
        augmentation_type: str = 'masked',
        difficulty: Union[DifficultyLevel, str] = DifficultyLevel.EASY,
        **kwargs
    ) -> BaseImageAugmentation:
        """
        Create an image augmentation strategy.
        
        Args:
            augmentation_type: Type of augmentation ('masked', 'none', etc.)
            difficulty: Difficulty level for curriculum learning
            **kwargs: Additional arguments passed to augmentation constructor
            
        Returns:
            Instance of BaseImageAugmentation
            
        Raises:
            ValueError: If augmentation_type is not registered
        """
        augmentation_type = augmentation_type.lower()
        
        if augmentation_type not in cls._IMAGE_AUGMENTATIONS:
            raise ValueError(
                f"Unknown image augmentation type: {augmentation_type}. "
                f"Available types: {list(cls._IMAGE_AUGMENTATIONS.keys())}"
            )
        
        augmentation_class = cls._IMAGE_AUGMENTATIONS[augmentation_type]
        return augmentation_class(difficulty=difficulty, **kwargs)
    
    @classmethod
    def create_text_augmentation(
        cls,
        augmentation_type: str = 'simple',
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        seed: Optional[int] = None
    ) -> BaseTextAugmentation:
        """
        Create a text augmentation instance.
        
        Args:
            augmentation_type: Type of text augmentation
                - 'simple': Basic word-level operations (language-agnostic)
                - 'rule-based': Rich Vietnamese linguistic rules
            difficulty: Difficulty level for curriculum learning
            seed: Random seed for reproducibility
            
        Returns:
            Text augmentation instance
        """
        if augmentation_type not in cls._text_augmentations:
            available = ', '.join(cls._text_augmentations.keys())
            raise ValueError(
                f"Unknown text augmentation type: {augmentation_type}. "
                f"Available types: {available}"
            )
        
        augmentation_class = cls._text_augmentations[augmentation_type]
        return augmentation_class(difficulty=difficulty, seed=seed)
    
    @classmethod
    def get_available_image_augmentations(cls) -> list:
        """Get list of registered image augmentation types."""
        return list(cls._IMAGE_AUGMENTATIONS.keys())
    
    @classmethod
    def get_available_text_augmentations(cls) -> list:
        """Get list of registered text augmentation types."""
        return list(cls._TEXT_AUGMENTATIONS.keys())
