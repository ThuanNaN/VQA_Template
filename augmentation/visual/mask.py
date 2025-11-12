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
from ..base import BaseImageAugmentation


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
        difficulty: Union[int, float] = 0.0,
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
        """Get default mask ratio based on difficulty level (0.0-1.0)."""
        # Smooth interpolation: 5% to 50% masking
        min_ratio = 0.05
        max_ratio = 0.50
        return min_ratio + self.difficulty * (max_ratio - min_ratio)
    
    def _configure_parameters(self):
        """Configure augmentation parameters based on difficulty level (0.0-1.0)."""
        # Set default mask ratio based on difficulty if not provided
        if self._mask_ratio is None:
            self.mask_ratio = self._get_default_mask_ratio()
        else:
            self.mask_ratio = self._mask_ratio
        
        # Smooth interpolation for all parameters based on difficulty (0.0-1.0)
        # Color jitter: 0.0 to 0.5
        self.color_jitter_strength = self.difficulty * 0.5
        
        # Brightness: (1.0, 1.0) to (0.5, 1.5)
        brightness_range = self.difficulty * 0.5
        self.brightness_factor = (1.0 - brightness_range, 1.0 + brightness_range)
        
        # Contrast: (1.0, 1.0) to (0.5, 1.5)
        contrast_range = self.difficulty * 0.5
        self.contrast_factor = (1.0 - contrast_range, 1.0 + contrast_range)
        
        # Blur radius: 0.0 to 3.0
        min_blur, max_blur = 0.0, 3.0
        blur_range = self.difficulty * max_blur
        self.blur_radius = (min_blur, blur_range)
        
        # Apply flip if difficulty > 0.3
        self.apply_flip = self.difficulty > 0.3
        
        # Crop scale: 1.0 to 0.7 (inverse relationship)
        min_scale = 0.7
        self.crop_scale = (min_scale + (1.0 - self.difficulty) * (1.0 - min_scale), 1.0)
    
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
