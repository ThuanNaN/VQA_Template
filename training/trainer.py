"""
Custom Trainer with Curriculum Learning support.

This module extends HuggingFace's Trainer to integrate curriculum learning
and dynamic augmentation strategies.
"""

from typing import Optional, Dict, Any, Callable, List
from transformers import Trainer, TrainerCallback
from augmentation import CurriculumScheduler
from utils.visualization import SampleObserver, WrongPredictionTracker
from PIL import Image
import torch
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


class SampleObservationCallback(TrainerCallback):
    """
    Callback for saving sample observations during training.
    
    At the end of each epoch, this callback:
    - Extracts sample data from the training set
    - Captures both original and augmented versions
    - Gets model predictions
    - Saves everything in an organized format
    
    Args:
        sample_observer: SampleObserver instance for saving data
        train_dataset: Training dataset
        num_samples: Number of samples to observe per epoch
    """
    
    def __init__(
        self,
        sample_observer: SampleObserver,
        train_dataset: Any,
        num_samples: int = 5
    ):
        self.sample_observer = sample_observer
        self.train_dataset = train_dataset
        self.num_samples = num_samples
        self.label_decoder = None
        
        # Create reverse mapping from label encoder
        if hasattr(train_dataset, 'label_encoder'):
            self.label_decoder = {v: k for k, v in train_dataset.label_encoder.items()}
    
    def on_epoch_end(self, args, state, control, model, **kwargs):
        """Save sample observations at the end of each epoch."""
        epoch = int(state.epoch) if state.epoch is not None else 0
        
        logger.info(f"Collecting sample observations for epoch {epoch}...")
        
        # Set model to eval mode
        model.eval()
        
        # Collect samples
        samples = []
        indices = torch.randperm(len(self.train_dataset))[:self.num_samples].tolist()
        
        with torch.no_grad():
            for idx in indices:
                try:
                    sample = self._collect_sample(idx, model, args.device)
                    samples.append(sample)
                except Exception as e:
                    logger.warning(f"Failed to collect sample {idx}: {e}")
                    continue
        
        # Save samples
        if samples and self.label_decoder:
            self.sample_observer.save_epoch_samples(
                epoch=epoch,
                samples=samples,
                label_decoder=self.label_decoder
            )
        
        # Set model back to train mode
        model.train()
    
    def _collect_sample(self, idx: int, model, device) -> Dict[str, Any]:
        """
        Collect a single sample with original and augmented versions.
        
        Args:
            idx: Sample index in dataset
            model: The VQA model
            device: Device to run inference on
            
        Returns:
            Dictionary containing sample data
        """
        # Get original data (temporarily disable augmentation)
        original_img_aug = self.train_dataset.image_augmentation
        original_text_aug = self.train_dataset.text_augmentation
        
        # Get original version
        self.train_dataset.set_image_augmentation(None)
        self.train_dataset.set_text_augmentation(None)
        
        # Load original image
        img_path = self.train_dataset.data['img_paths'][idx]
        original_image = Image.open(img_path).convert('RGB')
        original_question = self.train_dataset.data['questions'][idx]
        
        # Get original processed data
        original_data = self.train_dataset[idx]
        
        # Restore augmentation and get augmented version
        self.train_dataset.set_image_augmentation(original_img_aug)
        self.train_dataset.set_text_augmentation(original_text_aug)
        
        augmented_data = self.train_dataset[idx]
        
        # Get augmented versions if augmentation is enabled
        augmented_image = None
        augmented_question = None
        
        if original_img_aug is not None:
            augmented_image = original_img_aug(original_image)
        
        if original_text_aug is not None:
            augmented_question = original_text_aug(original_question)
        
        # Get model prediction on augmented data
        batch = {
            'image': augmented_data['image'].unsqueeze(0).to(device),
            'question_input_ids': augmented_data['question_input_ids'].unsqueeze(0).to(device),
            'question_attention_mask': augmented_data['question_attention_mask'].unsqueeze(0).to(device),
        }
        
        outputs = model(**batch)
        prediction = outputs["logits"].argmax(dim=-1).item()
        ground_truth = augmented_data['label'].item()
        
        return {
            'original_image': original_image,
            'augmented_image': augmented_image,
            'original_question': original_question,
            'augmented_question': augmented_question,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'img_path': img_path
        }


class BestMetricCallback(TrainerCallback):
    """
    Callback for tracking the best metric (accuracy) during training.
    
    This callback tracks the best accuracy achieved during validation
    and displays it at the end of training.
    """
    
    def __init__(self):
        self.best_accuracy = 0.0
        self.best_epoch = 0
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Update best accuracy after each evaluation."""
        if metrics is None:
            return
        
        # Check for accuracy metric
        accuracy = metrics.get('eval_accuracy', None)
        if accuracy is not None and accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = int(state.epoch) if state.epoch is not None else 0
            logger.info(f"New best accuracy: {self.best_accuracy:.4f} at epoch {self.best_epoch}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Display best accuracy at the end of training."""
        logger.info("=" * 80)
        logger.info(f"Training completed!")
        logger.info(f"Best Accuracy: {self.best_accuracy:.4f} (achieved at epoch {self.best_epoch})")
        logger.info("=" * 80)


class WrongPredictionCallback(TrainerCallback):
    """
    Callback for tracking all wrong predictions during validation.
    
    At the end of each validation epoch, this callback:
    - Collects all samples that were incorrectly predicted
    - Saves detailed metadata including images, questions, predictions, and ground truth
    - Creates summary reports for analysis
    
    Args:
        wrong_prediction_tracker: WrongPredictionTracker instance for saving data
        val_dataset: Validation dataset
    """
    
    def __init__(
        self,
        wrong_prediction_tracker: WrongPredictionTracker,
        val_dataset: Any
    ):
        self.tracker = wrong_prediction_tracker
        self.val_dataset = val_dataset
        self.label_decoder = None
        
        # Create reverse mapping from label encoder
        if hasattr(val_dataset, 'label_encoder'):
            self.label_decoder = {v: k for k, v in val_dataset.label_encoder.items()}
    
    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        """Track wrong predictions after each evaluation."""
        if metrics is None or self.label_decoder is None:
            return
        
        epoch = int(state.epoch) if state.epoch is not None else 0
        
        logger.info(f"Collecting wrong predictions for epoch {epoch}...")
        
        # Set model to eval mode
        model.eval()
        
        # Collect all wrong predictions
        wrong_samples = []
        
        with torch.no_grad():
            for idx in range(len(self.val_dataset)):
                try:
                    sample = self.val_dataset[idx]
                    
                    # Create batch
                    batch = {
                        'image': sample['image'].unsqueeze(0).to(args.device),
                        'question_input_ids': sample['question_input_ids'].unsqueeze(0).to(args.device),
                        'question_attention_mask': sample['question_attention_mask'].unsqueeze(0).to(args.device),
                    }
                    
                    # Get prediction
                    outputs = model(**batch)
                    logits = outputs["logits"][0]  # Remove batch dimension
                    prediction = logits.argmax(dim=-1).item()
                    ground_truth = sample['label'].item()
                    
                    # Track if prediction is wrong
                    if prediction != ground_truth:
                        # Get original data
                        img_path = self.val_dataset.data['img_paths'][idx]
                        question = self.val_dataset.data['questions'][idx]
                        
                        wrong_sample = {
                            'idx': idx,
                            'image_path': img_path,
                            'question': question,
                            'prediction': prediction,
                            'ground_truth': ground_truth,
                            'logits': logits
                        }
                        wrong_samples.append(wrong_sample)
                
                except Exception as e:
                    logger.warning(f"Failed to process sample {idx}: {e}")
                    continue
        
        # Save wrong predictions
        if wrong_samples:
            num_wrong = self.tracker.save_wrong_predictions(
                epoch=epoch,
                wrong_samples=wrong_samples,
                label_decoder=self.label_decoder
            )
            
            # Calculate error rate
            total_samples = len(self.val_dataset)
            error_rate = num_wrong / total_samples if total_samples > 0 else 0.0
            
            logger.info(
                f"Epoch {epoch}: {num_wrong}/{total_samples} wrong predictions "
                f"(error rate: {error_rate:.2%})"
            )
        else:
            logger.info(f"Epoch {epoch}: No wrong predictions! Perfect accuracy!")


class VQATrainer(Trainer):
    """
    Custom Trainer for VQA with Curriculum Learning support.
    
    Extends HuggingFace Trainer with additional features:
    - Curriculum learning integration
    - Dynamic augmentation updates
    - Sample observation logging
    - Wrong prediction tracking
    - VQA-specific logging and metrics
    
    Args:
        curriculum_scheduler: Optional CurriculumScheduler for curriculum learning
        augmentation_factory: Function that creates augmentation given difficulty value (0.0-1.0)
        sample_observer: Optional SampleObserver for saving sample observations
        num_observation_samples: Number of samples to observe per epoch
        wrong_prediction_tracker: Optional WrongPredictionTracker for tracking validation errors
        *args, **kwargs: Arguments passed to base Trainer
    """
    
    def __init__(
        self,
        curriculum_scheduler: Optional[CurriculumScheduler] = None,
        augmentation_factory: Optional[Callable[[float], Any]] = None,
        sample_observer: Optional[SampleObserver] = None,
        num_observation_samples: int = 5,
        wrong_prediction_tracker: Optional[WrongPredictionTracker] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.curriculum_scheduler = curriculum_scheduler
        self.augmentation_factory = augmentation_factory
        self.sample_observer = sample_observer
        self.num_observation_samples = num_observation_samples
        self.wrong_prediction_tracker = wrong_prediction_tracker
        
        # Add best metric tracking callback (always enabled)
        self._setup_best_metric_tracking()
        
        # Add curriculum learning callback if provided
        if curriculum_scheduler is not None and augmentation_factory is not None:
            self._setup_curriculum_learning()
        
        # Add sample observation callback if provided
        if sample_observer is not None:
            self._setup_sample_observation()
        
        # Add wrong prediction tracking callback if provided
        if wrong_prediction_tracker is not None:
            self._setup_wrong_prediction_tracking()
    
    def _setup_best_metric_tracking(self):
        """Set up best metric tracking callback."""
        best_metric_callback = BestMetricCallback()
        self.add_callback(best_metric_callback)
        logger.info("Best Metric Tracking enabled")
    
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
    
    def _setup_sample_observation(self):
        """Set up sample observation callback."""
        observation_callback = SampleObservationCallback(
            sample_observer=self.sample_observer,
            train_dataset=self.train_dataset,
            num_samples=self.num_observation_samples
        )
        
        self.add_callback(observation_callback)
        logger.info(f"Sample Observation enabled: {self.num_observation_samples} samples per epoch")
    
    def _setup_wrong_prediction_tracking(self):
        """Set up wrong prediction tracking callback."""
        wrong_pred_callback = WrongPredictionCallback(
            wrong_prediction_tracker=self.wrong_prediction_tracker,
            val_dataset=self.eval_dataset
        )
        
        self.add_callback(wrong_pred_callback)
        logger.info("Wrong Prediction Tracking enabled for validation set")

    
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
