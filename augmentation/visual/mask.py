"""
Image augmentation for Visual Question Answering with Curriculum Learning.

Inspired by: "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)
Paper: https://arxiv.org/abs/2111.06377

This module implements image augmentation techniques with a focus on 
Curriculum Learning (CL) to guide model learning from easy to hard samples.
The framework provides flexible difficulty levels for progressive training.
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import Optional, Union
from ..base import BaseImageAugmentation


class MaskedImageAugmentation(BaseImageAugmentation):
    """
    Image augmentation framework with Curriculum Learning support.
    
    Implements various augmentation techniques inspired by Masked Autoencoders,
    organized by difficulty levels to enable curriculum learning from easy to hard.
    
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
        # Smooth interpolation: 15% to 50% masking
        min_ratio = 0.15
        max_ratio = 0.50
        return min_ratio + self.difficulty * (max_ratio - min_ratio)
    
    def _configure_parameters(self):
        """Configure augmentation parameters based on difficulty level (0.0-1.0).
        
        Three-phase strategy:
        - Easy (0.0 - 0.33): No augmentation
        - Medium (0.33 - 0.66): Blur random blocks
        - Hard (0.66 - 1.0): Mask random blocks
        """
        # Set default mask ratio based on difficulty if not provided
        if self._mask_ratio is None:
            self.mask_ratio = self._get_default_mask_ratio()
        else:
            self.mask_ratio = self._mask_ratio
        
        # Determine augmentation phase
        if self.difficulty < 0.33:
            # Easy: No augmentation
            self.augmentation_phase = 'easy'
            self.apply_blur_blocks = False
            self.apply_mask_blocks = False
        elif self.difficulty < 0.66:
            # Medium: Blur random blocks
            self.augmentation_phase = 'medium'
            self.apply_blur_blocks = True
            self.apply_mask_blocks = False
        else:
            # Hard: Mask random blocks
            self.augmentation_phase = 'hard'
            self.apply_blur_blocks = False
            self.apply_mask_blocks = True
        
        # Block blur radius for medium phase: 2.0 to 5.0
        self.block_blur_radius = 3.5
    
    def augment(
        self, 
        image: Image.Image,
        apply_masking: bool = True,
    ) -> Image.Image:
        """
        Apply augmentation to an image based on difficulty level.
        
        Three-phase strategy:
        - Easy (difficulty < 0.33): No augmentation, return original
        - Medium (0.33 <= difficulty < 0.66): Blur random blocks
        - Hard (difficulty >= 0.66): Mask random blocks
        
        Args:
            image: PIL Image to augment
            apply_masking: Whether to apply random masking (hard phase)
            
        Returns:
            List containing single augmented PIL Image.
            Returns [original_image] for easy phase (difficulty < 0.33).
        """
        # Easy phase: No augmentation
        if self.augmentation_phase == 'easy':
            return image
        
        augmented_image = image.copy()
        
        # Medium phase: Blur random blocks
        if self.augmentation_phase == 'medium' and self.apply_blur_blocks:
            augmented_image = self._blur_random_blocks(augmented_image)
        
        # Hard phase: Mask random blocks
        if self.augmentation_phase == 'hard' and self.apply_mask_blocks and apply_masking:
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
    
    def _blur_random_blocks(self, image: Image.Image) -> Image.Image:
        """
        Apply Gaussian blur to random blocks (patches) of the image.
        Used in medium difficulty phase.
        
        Args:
            image: PIL Image to blur
            
        Returns:
            Image with random blocks blurred
        """
        width, height = image.size
        
        # Convert to numpy for easier manipulation
        img_array = np.array(image)
        
        # Calculate number of patches
        n_patches_h = height // self.patch_size
        n_patches_w = width // self.patch_size
        total_patches = n_patches_h * n_patches_w
        
        # Determine number of patches to blur (same ratio as masking)
        n_blurred = int(total_patches * self.mask_ratio)
        
        # Create blur indices
        patch_indices = np.arange(total_patches)
        blurred_indices = self.rng.choice(patch_indices, size=n_blurred, replace=False)
        
        # Convert back to PIL for filtering
        pil_image = Image.fromarray(img_array)
        
        # Apply blur to selected patches
        for idx in blurred_indices:
            # Convert 1D index to 2D patch coordinates
            patch_row = idx // n_patches_w
            patch_col = idx % n_patches_w
            
            # Calculate pixel coordinates
            y_start = patch_row * self.patch_size
            y_end = min(y_start + self.patch_size, height)
            x_start = patch_col * self.patch_size
            x_end = min(x_start + self.patch_size, width)
            
            # Extract patch, blur it, and paste back
            patch = pil_image.crop((x_start, y_start, x_end, y_end))
            blurred_patch = patch.filter(ImageFilter.GaussianBlur(radius=self.block_blur_radius))
            pil_image.paste(blurred_patch, (x_start, y_start))
        
        return pil_image
    
    
    def get_augmentation_info(self) -> dict:
        """
        Get information about current augmentation configuration.
        
        Returns:
            Dictionary with augmentation parameters
        """
        return {
            "difficulty": self.difficulty,
            "augmentation_phase": self.augmentation_phase,
            "patch_size": self.patch_size,
            "mask_ratio": self.mask_ratio,
            "apply_blur_blocks": self.apply_blur_blocks,
            "apply_mask_blocks": self.apply_mask_blocks,
            "block_blur_radius": getattr(self, 'block_blur_radius', None),
        }
