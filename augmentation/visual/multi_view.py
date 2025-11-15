"""
Multi-view image augmentation for VQA.

This augmentation generates multiple views/transformations of the input image
and returns them as a list for ensemble encoding.
"""

from typing import List
from PIL import Image
import random
from ..base import BaseImageAugmentation


class MultiViewImageAugmentation(BaseImageAugmentation):
    """
    Generate multiple augmented views of the same image for ensemble encoding.
    
    Creates N different augmented versions of the input image that will be
    encoded separately and then aggregated (mean/sum/max pooling).
    """
    
    def __init__(self, difficulty: float = 0.5, num_views: int = 3):
        """
        Initialize multi-view image augmentation.
        
        Args:
            difficulty: Controls augmentation strength (0.0 to 1.0)
            num_views: Number of augmented views to generate
        """
        super().__init__(difficulty)
        self.num_views = num_views
        self._configure_parameters()
    
    def _configure_parameters(self):
        """Configure augmentation parameters based on difficulty."""
        # Rotation range increases with difficulty
        self.rotation_range = int(5 + self.difficulty * 15)  # 5 to 20 degrees
        
        # Brightness/contrast variation
        self.brightness_range = (0.9 - self.difficulty * 0.2, 
                                1.1 + self.difficulty * 0.2)  # (0.7-1.3) to (0.9-1.1)
        
        # Color jitter
        self.color_jitter = 0.1 + self.difficulty * 0.3  # 0.1 to 0.4
    
    def augment(self, image: Image.Image) -> List[Image.Image]:
        """
        Generate multiple augmented views of the image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            List of augmented PIL Images (including original)
        """
        from PIL import ImageEnhance
        
        views = [image.copy()]  # Always include original
        
        for _ in range(self.num_views - 1):
            aug_image = image.copy()
            
            # Apply random rotation
            if random.random() < 0.5:
                angle = random.uniform(-self.rotation_range, self.rotation_range)
                aug_image = aug_image.rotate(angle, expand=False, fillcolor=(0, 0, 0))
            
            # Apply brightness adjustment
            if random.random() < 0.5:
                brightness = random.uniform(*self.brightness_range)
                enhancer = ImageEnhance.Brightness(aug_image)
                aug_image = enhancer.enhance(brightness)
            
            # Apply contrast adjustment
            if random.random() < 0.5:
                contrast = random.uniform(*self.brightness_range)
                enhancer = ImageEnhance.Contrast(aug_image)
                aug_image = enhancer.enhance(contrast)
            
            # Apply color jitter
            if random.random() < 0.5:
                color = random.uniform(1 - self.color_jitter, 1 + self.color_jitter)
                enhancer = ImageEnhance.Color(aug_image)
                aug_image = enhancer.enhance(color)
            
            views.append(aug_image)
        
        return views


class CropMultiViewAugmentation(BaseImageAugmentation):
    """
    Generate multiple crops of the image for multi-scale feature extraction.
    
    Creates different crops (center, random, corners) to capture various
    parts of the image, useful for focusing on different regions.
    """
    
    def __init__(self, difficulty: float = 0.5, num_views: int = 3):
        """
        Initialize crop-based multi-view augmentation.
        
        Args:
            difficulty: Controls crop size (higher = smaller crops)
            num_views: Number of different crops to generate
        """
        super().__init__(difficulty)
        self.num_views = num_views
        self._configure_parameters()
    
    def _configure_parameters(self):
        """Configure crop size based on difficulty."""
        # Higher difficulty = smaller crops (more zoomed in)
        self.crop_scale = 1.0 - (self.difficulty * 0.3)  # 0.7 to 1.0
    
    def augment(self, image: Image.Image) -> List[Image.Image]:
        """
        Generate multiple crops of the image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            List of cropped PIL Images
        """
        views = []
        width, height = image.size
        crop_width = int(width * self.crop_scale)
        crop_height = int(height * self.crop_scale)
        
        # Always include center crop
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        center_crop = image.crop((left, top, left + crop_width, top + crop_height))
        center_crop = center_crop.resize((width, height), Image.Resampling.LANCZOS)
        views.append(center_crop)
        
        # Generate additional random/corner crops
        crop_positions = [
            (0, 0),  # Top-left
            (width - crop_width, 0),  # Top-right
            (0, height - crop_height),  # Bottom-left
            (width - crop_width, height - crop_height),  # Bottom-right
        ]
        
        for _ in range(self.num_views - 1):
            if random.random() < 0.5 and crop_positions:
                # Use corner crop
                left, top = crop_positions.pop(random.randrange(len(crop_positions)))
            else:
                # Random crop
                left = random.randint(0, width - crop_width)
                top = random.randint(0, height - crop_height)
            
            crop = image.crop((left, top, left + crop_width, top + crop_height))
            crop = crop.resize((width, height), Image.Resampling.LANCZOS)
            views.append(crop)
        
        return views
