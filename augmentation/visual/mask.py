"""
Image augmentation for Visual Question Answering with Curriculum Learning.

Inspired by: "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)
Paper: https://arxiv.org/abs/2111.06377

This module implements image augmentation techniques with a focus on 
Curriculum Learning (CL) to guide model learning from easy to hard samples.
The framework provides flexible difficulty levels for progressive training.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Optional, Union
from ..base import (
    BaseImageAugmentation, 
    DifficultyLevel, 
)
from ..scheduler import CurriculumLearningScheduler


class MaskedImageAugmentation(BaseImageAugmentation):
    """
    Image augmentation framework with Curriculum Learning support.
    
    Implements various augmentation techniques inspired by Masked Autoencoders,
    organized by difficulty levels to enable curriculum learning from easy to hard.
    
    Key Features:
    - Random masking (MAE-inspired): Patch-based and block masking
    - Color jittering with varying intensities
    - Gaussian blur with adjustable strengths
    - Brightness and contrast adjustments
    - Random cropping and flipping
    - Curriculum Learning support (easy → medium → hard progression)
    
    Args:
        difficulty: Difficulty level for augmentation (EASY, MEDIUM, or HARD)
        patch_size: Size of patches for masked autoencoding (default: 16)
        mask_ratio: Ratio of patches to mask (default varies by difficulty)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        difficulty: Union[DifficultyLevel, str] = DifficultyLevel.EASY,
        patch_size: int = 16,
        mask_ratio: Optional[float] = None,
        seed: Optional[int] = None
    ):
        """Initialize the augmentation framework."""
        self.patch_size = patch_size
        self._mask_ratio = mask_ratio
        
        # Create random number generator for reproducibility
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self._random_state_counter = 0
        else:
            self.rng = np.random.RandomState()
        
        # Call parent constructor which will call _configure_parameters
        super().__init__(difficulty=difficulty, seed=seed)
    
    def _get_default_mask_ratio(self) -> float:
        """Get default mask ratio based on difficulty level."""
        if self.difficulty == DifficultyLevel.EASY:
            return 0.15  # 15% masking for easy samples
        elif self.difficulty == DifficultyLevel.MEDIUM:
            return 0.50  # 50% masking for medium samples
        else:  # HARD
            return 0.75  # 75% masking for hard samples (MAE paper uses 75%)
    
    def _configure_parameters(self):
        """Configure augmentation parameters based on difficulty level."""
        # Set default mask ratio based on difficulty if not provided
        if self._mask_ratio is None:
            self.mask_ratio = self._get_default_mask_ratio()
        else:
            self.mask_ratio = self._mask_ratio
        
        if self.difficulty == DifficultyLevel.EASY:
            # Easy: Minimal transformations
            self.color_jitter_strength = 0.1
            self.brightness_factor = (0.9, 1.1)
            self.contrast_factor = (0.9, 1.1)
            self.blur_radius = (0.1, 0.3)
            self.apply_flip = False
            self.crop_scale = (0.95, 1.0)
            
        elif self.difficulty == DifficultyLevel.MEDIUM:
            # Medium: Moderate transformations
            self.color_jitter_strength = 0.3
            self.brightness_factor = (0.7, 1.3)
            self.contrast_factor = (0.7, 1.3)
            self.blur_radius = (0.5, 1.5)
            self.apply_flip = True
            self.crop_scale = (0.8, 1.0)
            
        else:  # HARD
            # Hard: Aggressive transformations
            self.color_jitter_strength = 0.5
            self.brightness_factor = (0.5, 1.5)
            self.contrast_factor = (0.5, 1.5)
            self.blur_radius = (1.0, 3.0)
            self.apply_flip = True
            self.crop_scale = (0.7, 1.0)
    
    def augment(
        self, 
        image: Image.Image,
        apply_masking: bool = True,
        apply_color_jitter: bool = True,
        apply_blur: bool = True,
        apply_brightness: bool = True,
        apply_contrast: bool = True,
        apply_crop: bool = False,
        apply_flip: bool = None
    ) -> Image.Image:
        """
        Apply augmentation to an image based on difficulty level.
        
        Args:
            image: PIL Image to augment
            apply_masking: Whether to apply random masking
            apply_color_jitter: Whether to apply color jittering
            apply_blur: Whether to apply Gaussian blur
            apply_brightness: Whether to adjust brightness
            apply_contrast: Whether to adjust contrast
            apply_crop: Whether to apply random cropping
            apply_flip: Whether to apply random horizontal flip (uses difficulty default if None)
            
        Returns:
            Augmented PIL Image
        """
        augmented_image = image.copy()
        
        # Apply transformations in order
        if apply_crop:
            augmented_image = self._random_crop(augmented_image)
        
        if apply_flip is None:
            apply_flip = self.apply_flip
        if apply_flip and self.rng.random() > 0.5:
            augmented_image = augmented_image.transpose(Image.FLIP_LEFT_RIGHT)
        
        if apply_color_jitter:
            augmented_image = self._color_jitter(augmented_image)
        
        if apply_brightness:
            augmented_image = self._adjust_brightness(augmented_image)
        
        if apply_contrast:
            augmented_image = self._adjust_contrast(augmented_image)
        
        if apply_blur:
            augmented_image = self._gaussian_blur(augmented_image)
        
        if apply_masking:
            augmented_image = self._random_masking(augmented_image)
        
        return augmented_image
    
    def _random_masking(self, image: Image.Image) -> Image.Image:
        """
        Apply random masking inspired by Masked Autoencoders.
        
        Randomly masks patches of the image with gray color (128, 128, 128).
        The masking strategy varies by difficulty level.
        
        Args:
            image: PIL Image to mask
            
        Returns:
            Masked PIL Image
        """
        width, height = image.size
        
        # Convert to numpy for easier manipulation
        img_array = np.array(image)
        
        # Calculate number of patches
        n_patches_h = height // self.patch_size
        n_patches_w = width // self.patch_size
        total_patches = n_patches_h * n_patches_w
        
        # Determine number of patches to mask
        n_masked = int(total_patches * self.mask_ratio)
        
        # Create mask indices
        patch_indices = np.arange(total_patches)
        masked_indices = self.rng.choice(patch_indices, size=n_masked, replace=False)
        
        # Apply masking
        mask_color = 128  # Gray color for masked regions
        
        for idx in masked_indices:
            # Convert 1D index to 2D patch coordinates
            patch_row = idx // n_patches_w
            patch_col = idx % n_patches_w
            
            # Calculate pixel coordinates
            y_start = patch_row * self.patch_size
            y_end = min(y_start + self.patch_size, height)
            x_start = patch_col * self.patch_size
            x_end = min(x_start + self.patch_size, width)
            
            # Mask the patch
            img_array[y_start:y_end, x_start:x_end] = mask_color
        
        return Image.fromarray(img_array)
    
    def _color_jitter(self, image: Image.Image) -> Image.Image:
        """
        Apply color jittering to the image.
        
        Args:
            image: PIL Image to transform
            
        Returns:
            Color-jittered PIL Image
        """
        # Random hue shift
        enhancer = ImageEnhance.Color(image)
        factor = 1 + self.rng.uniform(-self.color_jitter_strength, self.color_jitter_strength)
        image = enhancer.enhance(factor)
        
        return image
    
    def _adjust_brightness(self, image: Image.Image) -> Image.Image:
        """
        Adjust image brightness.
        
        Args:
            image: PIL Image to transform
            
        Returns:
            Brightness-adjusted PIL Image
        """
        enhancer = ImageEnhance.Brightness(image)
        factor = self.rng.uniform(self.brightness_factor[0], self.brightness_factor[1])
        return enhancer.enhance(factor)
    
    def _adjust_contrast(self, image: Image.Image) -> Image.Image:
        """
        Adjust image contrast.
        
        Args:
            image: PIL Image to transform
            
        Returns:
            Contrast-adjusted PIL Image
        """
        enhancer = ImageEnhance.Contrast(image)
        factor = self.rng.uniform(self.contrast_factor[0], self.contrast_factor[1])
        return enhancer.enhance(factor)
    
    def _gaussian_blur(self, image: Image.Image) -> Image.Image:
        """
        Apply Gaussian blur to the image.
        
        Args:
            image: PIL Image to blur
            
        Returns:
            Blurred PIL Image
        """
        radius = self.rng.uniform(self.blur_radius[0], self.blur_radius[1])
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _random_crop(self, image: Image.Image) -> Image.Image:
        """
        Apply random cropping and resize back to original size.
        
        Args:
            image: PIL Image to crop
            
        Returns:
            Cropped and resized PIL Image
        """
        width, height = image.size
        scale = self.rng.uniform(self.crop_scale[0], self.crop_scale[1])
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Random crop position
        left = self.rng.randint(0, width - new_width + 1)
        top = self.rng.randint(0, height - new_height + 1)
        
        cropped = image.crop((left, top, left + new_width, top + new_height))
        # Resize back to original size
        return cropped.resize((width, height), Image.BILINEAR)
    
    def get_augmentation_info(self) -> dict:
        """
        Get information about current augmentation configuration.
        
        Returns:
            Dictionary with augmentation parameters
        """
        return {
            "difficulty": self.difficulty.value,
            "patch_size": self.patch_size,
            "mask_ratio": self.mask_ratio,
            "color_jitter_strength": self.color_jitter_strength,
            "brightness_factor": self.brightness_factor,
            "contrast_factor": self.contrast_factor,
            "blur_radius": self.blur_radius,
            "apply_flip": self.apply_flip,
            "crop_scale": self.crop_scale,
        }


def create_augmentor_for_epoch(
    epoch: int,
    scheduler: CurriculumLearningScheduler,
    patch_size: int = 16,
    seed: Optional[int] = None
) -> MaskedImageAugmentation:
    """
    Create an image augmentor configured for the current epoch.
    
    Convenience function to create an augmentor with the appropriate
    difficulty level based on the curriculum learning schedule.
    
    Args:
        epoch: Current epoch number
        scheduler: CurriculumLearningScheduler instance
        patch_size: Size of patches for masking
        seed: Random seed for reproducibility
        
    Returns:
        MaskedImageAugmentation instance configured for the epoch
    """
    difficulty = scheduler.get_difficulty_for_epoch(epoch)
    return MaskedImageAugmentation(
        difficulty=difficulty,
        patch_size=patch_size,
        seed=seed
    )

