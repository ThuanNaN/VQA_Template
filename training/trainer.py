"""
Custom Trainer with Curriculum Learning support.

This module extends HuggingFace's Trainer to integrate curriculum learning
and dynamic augmentation strategies.
"""

from typing import Optional, Dict, Any, Callable
from transformers import Trainer, TrainerCallback
from augmentation import CurriculumScheduler
import logging

logger = logging.getLogger(__name__)


class CurriculumLearningCallback(TrainerCallback):
    """
    Callback for managing curriculum learning progression during training.
    
    This callback updates dataset augmentation strategies based on the
    current epoch and curriculum schedule.
    
    Args:
        scheduler: CurriculumScheduler instance
        train_dataset: Training dataset with set_image_augmentation method
        augmentation_factory: Function that creates augmentation given difficulty value (0.0-1.0)
        val_dataset: Optional validation dataset
    """
    
    def __init__(
        self,
        scheduler: CurriculumScheduler,
        train_dataset: Any,
        augmentation_factory: Callable[[float], Any],
        val_dataset: Optional[Any] = None
    ):
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.augmentation_factory = augmentation_factory
        self.current_difficulty = None
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Update augmentation difficulty at the beginning of each epoch."""
        epoch = int(state.epoch) if state.epoch is not None else 0
        
        # Get difficulty for current epoch (0.0 to 1.0)
        difficulty = self.scheduler.get_difficulty(epoch)
        
        # Only update if difficulty changed significantly (threshold to avoid too frequent updates)
        if self.current_difficulty is None or abs(difficulty - self.current_difficulty) > 0.01:
            self.current_difficulty = difficulty
            
            logger.info(f"Epoch {epoch}: Updating curriculum difficulty to {difficulty:.3f}")
            
            # Create new augmentations with current difficulty
            augmentors = self.augmentation_factory(difficulty)
            
            # Update datasets
            if hasattr(self.train_dataset, 'set_image_augmentation'):
                # Update image augmentation if present
                if 'image' in augmentors:
                    image_augmentor = augmentors['image']
                    augment_fn = lambda img: image_augmentor.augment(img)
                    self.train_dataset.set_image_augmentation(augment_fn)
                    logger.info(f"Updated image augmentation: {image_augmentor.get_augmentation_info()}")
                
                # Update text augmentation if present
                if 'text' in augmentors:
                    text_augmentor = augmentors['text']
                    augment_fn = lambda text: text_augmentor.augment(text)
                    self.train_dataset.set_text_augmentation(augment_fn)
                    logger.info(f"Updated text augmentation: {text_augmentor.get_augmentation_info()}")
            
            # Optionally disable augmentation for validation
            if self.val_dataset is not None:
                if hasattr(self.val_dataset, 'set_image_augmentation'):
                    self.val_dataset.set_image_augmentation(None)
                if hasattr(self.val_dataset, 'set_text_augmentation'):
                    self.val_dataset.set_text_augmentation(None)


class VQATrainer(Trainer):
    """
    Custom Trainer for VQA with Curriculum Learning support.
    
    Extends HuggingFace Trainer with additional features:
    - Curriculum learning integration
    - Dynamic augmentation updates
    - VQA-specific logging and metrics
    
    Args:
        curriculum_scheduler: Optional CurriculumScheduler for curriculum learning
        augmentation_factory: Function that creates augmentation given difficulty value (0.0-1.0)
        *args, **kwargs: Arguments passed to base Trainer
    """
    
    def __init__(
        self,
        curriculum_scheduler: Optional[CurriculumScheduler] = None,
        augmentation_factory: Optional[Callable[[float], Any]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.curriculum_scheduler = curriculum_scheduler
        self.augmentation_factory = augmentation_factory
        
        # Add curriculum learning callback if provided
        if curriculum_scheduler is not None and augmentation_factory is not None:
            self._setup_curriculum_learning()
    
    def _setup_curriculum_learning(self):
        """Set up curriculum learning callback."""
        curriculum_callback = CurriculumLearningCallback(
            scheduler=self.curriculum_scheduler,
            train_dataset=self.train_dataset,
            augmentation_factory=self.augmentation_factory,
            val_dataset=self.eval_dataset
        )
        
        self.add_callback(curriculum_callback)
        logger.info("Curriculum Learning enabled")
        logger.info(f"Schedule: {self.curriculum_scheduler.get_schedule_info()}")
    
    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        """
        Enhanced logging with curriculum learning info.
        
        Args:
            logs: Dictionary of metrics to log
            start_time: Optional start time for logging (passed by Trainer)
        """
        # Add curriculum difficulty to logs if available
        if self.curriculum_scheduler is not None and self.state.epoch is not None:
            epoch = int(self.state.epoch)
            difficulty = self.curriculum_scheduler.get_difficulty(epoch)
            logs['curriculum_difficulty'] = difficulty
        
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
