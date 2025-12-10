"""
Training pipeline for VQA models with curriculum learning support.

This module provides a high-level pipeline for training VQA models
with integrated augmentation and curriculum learning.
"""

import os
import torch
import wandb
from pathlib import Path
from typing import Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    EarlyStoppingCallback,
)

from .config import ExperimentConfig
from .trainer import VQATrainer
from dataset import (
    ViVQADataset,
    ViVQAAddonDataset,
    OpenViVQADataset,
    ViTextVQADataset,
    EVJVQADataset,
    ViVQAXDataset,
    ViVQAXAddonDataset,
    ViOCRVQADataset,
    ViOCRVQAAddonDataset,
)
from models import SimpleVQAConfig, SimpleVQA
from augmentation import (
    AugmentationFactory,
    CurriculumScheduler,
)
from utils import compute_metrics, seed_everything
from utils.visualization import create_sample_observer, create_wrong_prediction_tracker
import logging

logger = logging.getLogger(__name__)


class DatasetFactory:
    """Factory for creating VQA datasets."""
    
    DATASET_CLASSES = {
        'vivqa': ViVQADataset,
        'vivqa-addon': ViVQAAddonDataset,
        'openvivqa': OpenViVQADataset,
        'vitextvqa': ViTextVQADataset,
        'evjvqa': EVJVQADataset,
        'vivqax': ViVQAXDataset,
        'vivqax-addon': ViVQAXAddonDataset,
        'viocrvqa': ViOCRVQADataset,
        'viocrvqa-addon': ViOCRVQAAddonDataset,
    }
    
    @classmethod
    def create_dataset(cls, dataset_name: str, ann_path: str, img_dir: str,
                      text_processor, vis_processor, **kwargs):
        """Create a dataset instance."""
        if dataset_name not in cls.DATASET_CLASSES:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(cls.DATASET_CLASSES.keys())}"
            )
        
        dataset_class = cls.DATASET_CLASSES[dataset_name]
        return dataset_class(
            ann_path=ann_path,
            img_dir=img_dir,
            text_processor=text_processor,
            vis_processor=vis_processor,
            **kwargs
        )


class VQATrainingPipeline:
    """
    Complete training pipeline for VQA models.
    
    This pipeline handles:
    - Configuration management
    - Dataset creation with augmentation
    - Model initialization
    - Curriculum learning setup
    - Training with WandB logging
    
    Example:
        >>> config = ExperimentConfig.from_args(args)
        >>> pipeline = VQATrainingPipeline(config)
        >>> pipeline.run()
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the training pipeline.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.setup_environment()
        
    def setup_environment(self):
        """Set up random seeds and system threads."""
        seed_everything(self.config.training.seed)
        
        # Setup torch threads
        system_threads = torch.get_num_threads()
        running_threads = min(self.config.training.n_threads, system_threads)
        torch.set_num_threads(running_threads)
        torch.set_num_interop_threads(running_threads)
        
        logger.info(f"Environment setup complete (seed={self.config.training.seed}, threads={running_threads})")
    
    def setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if self.config.training.report_to_wandb:
            run_name = f"{self.config.data.dataset_name}-{self.config.training.run_name}-{self.config.training.seed}"
            
            wandb.init(
                project=self.config.training.wandb_project,
                name=run_name,
                config=self.config.to_dict()
            )
            logger.info(f"WandB initialized: {run_name}")
    
    def create_processors(self):
        """Create text and vision processors."""
        vis_processor = AutoProcessor.from_pretrained(
            self.config.model.vis_model_name,
            use_fast=True
        )
        text_processor = AutoTokenizer.from_pretrained(
            self.config.model.text_model_name
        )
        
        logger.info(f"Processors created: {self.config.model.vis_model_name}, {self.config.model.text_model_name}")
        return text_processor, vis_processor
    
    def create_datasets(self, text_processor, vis_processor) -> Tuple:
        """Create training and validation datasets."""
        # Create datasets without augmentation first
        train_dataset = DatasetFactory.create_dataset(
            dataset_name=self.config.data.dataset_name,
            ann_path=self.config.data.train_ann_path,
            img_dir=self.config.data.train_img_dir,
            text_processor=text_processor,
            vis_processor=vis_processor,
            max_length=self.config.data.seq_len
        )
        
        val_dataset = DatasetFactory.create_dataset(
            dataset_name=self.config.data.dataset_name,
            ann_path=self.config.data.val_ann_path,
            img_dir=self.config.data.val_img_dir,
            text_processor=text_processor,
            vis_processor=vis_processor,
            max_length=self.config.data.seq_len
        )
        
        logger.info(f"Datasets created: train={len(train_dataset)}, val={len(val_dataset)}")
        
        # Apply initial augmentation if enabled but not using curriculum
        if not self.config.augmentation.enable_curriculum:
            # Image augmentation
            if self.config.augmentation.enable_image_augmentation:
                image_augmentor = AugmentationFactory.create_image_augmentation(
                    augmentation_type=self.config.augmentation.image_augmentation_type,
                    difficulty=0.0,  # Start with minimum difficulty
                    patch_size=self.config.augmentation.patch_size,
                    seed=self.config.augmentation.seed
                )
                train_dataset.set_image_augmentation(lambda img: image_augmentor.augment(img))
                logger.info(f"Applied image augmentation: {image_augmentor.get_augmentation_info()}")
            
            # Text augmentation
            if self.config.augmentation.enable_text_augmentation:
                text_augmentor = AugmentationFactory.create_text_augmentation(
                    augmentation_type=self.config.augmentation.text_augmentation_type,
                    difficulty=0.0,  # Start with minimum difficulty
                    seed=self.config.augmentation.seed
                )
                train_dataset.set_text_augmentation(lambda text: text_augmentor.augment(text))
                logger.info(f"Applied text augmentation: {text_augmentor.get_augmentation_info()}")
        
        return train_dataset, val_dataset
    
    def create_model(self, num_classes: int):
        """Create VQA model."""
        config = SimpleVQAConfig(
            vis_model_name=self.config.model.vis_model_name,
            text_model_name=self.config.model.text_model_name,
            num_classes=num_classes,
            hidden_size=self.config.model.hidden_size,
            text_aggregation=self.config.model.text_aggregation,
            vis_aggregation=self.config.model.vis_aggregation,
            text_aggregation_kwargs=self.config.model.text_aggregation_kwargs,
            vis_aggregation_kwargs=self.config.model.vis_aggregation_kwargs,
        )
        model = SimpleVQA(config)
        
        logger.info(f"Model created with {num_classes} classes")
        logger.info(f"Text aggregation: {self.config.model.text_aggregation}")
        logger.info(f"Visual aggregation: {self.config.model.vis_aggregation}")
        return model, config
    
    def create_curriculum_scheduler(self) -> Optional[CurriculumScheduler]:
        """Create curriculum learning scheduler if enabled."""
        if not self.config.augmentation.enable_curriculum:
            return None
        
        # Create scheduler with configured strategy
        total_epochs = self.config.training.epochs
        config = self.config.augmentation
        
        # Prepare kwargs based on strategy
        scheduler_kwargs = {
            'total_epochs': total_epochs,
            'strategy': config.curriculum_strategy,
            'warmup_epochs': config.warmup_epochs,
        }
        
        # Add strategy-specific parameters
        if config.curriculum_strategy == 'exponential':
            scheduler_kwargs['gamma'] = config.curriculum_gamma
        elif config.curriculum_strategy == 'step':
            step_size = config.curriculum_step_size or (total_epochs // 3)
            scheduler_kwargs['step_size'] = step_size
            scheduler_kwargs['gamma'] = config.curriculum_gamma
        elif config.curriculum_strategy == 'polynomial':
            scheduler_kwargs['power'] = config.curriculum_power
        
        scheduler = CurriculumScheduler(**scheduler_kwargs)
        
        logger.info(f"Curriculum Learning enabled with {config.curriculum_strategy} strategy")
        logger.info(f"Schedule: {scheduler.get_schedule_info()}")
        
        return scheduler
    
    def create_augmentation_factory(self):
        """Create augmentation factory function for curriculum learning."""
        config = self.config.augmentation
        
        if not config.enable_image_augmentation and not config.enable_text_augmentation:
            return None
        
        def factory(difficulty: float):
            """Create augmentation strategies for given difficulty (0.0-1.0)."""
            augmentors = {}
            
            if config.enable_image_augmentation:
                augmentors['image'] = AugmentationFactory.create_image_augmentation(
                    augmentation_type=config.image_augmentation_type,
                    difficulty=difficulty,  # Now uses float 0.0-1.0
                    patch_size=config.patch_size,
                    seed=config.seed
                )
            
            if config.enable_text_augmentation:
                augmentors['text'] = AugmentationFactory.create_text_augmentation(
                    augmentation_type=config.text_augmentation_type,
                    difficulty=difficulty,  # Now uses float 0.0-1.0
                    seed=config.seed
                )
            
            return augmentors
        
        return factory
    
    def create_training_arguments(self, save_dir: Path) -> TrainingArguments:
        """Create HuggingFace TrainingArguments."""
        # Determine save format
        save_safetensors = True
        if self.config.model.text_model_name in ['vinai/bartpho-syllable-base', 'vinai/bartpho-syllable']:
            save_safetensors = False
        
        run_name = f"{self.config.data.dataset_name}-{self.config.training.run_name}-{self.config.training.seed}"
        
        return TrainingArguments(
            seed=self.config.training.seed,
            output_dir=str(save_dir),
            per_device_train_batch_size=self.config.data.batch_size,
            per_device_eval_batch_size=self.config.data.batch_size,
            dataloader_num_workers=self.config.data.dataloader_workers,
            dataloader_pin_memory=True,
            torch_compile=True,
            num_train_epochs=self.config.training.epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim="adamw_torch",
            gradient_accumulation_steps=self.config.training.gradient_accumulation,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            fp16=self.config.training.fp16,
            lr_scheduler_type="cosine",
            warmup_steps=self.config.training.warmup_steps,
            logging_dir="./logs",
            logging_steps=self.config.training.logging_steps,
            save_total_limit=1,
            push_to_hub=False,
            save_safetensors=save_safetensors,
            run_name=run_name,
            report_to="wandb" if self.config.training.report_to_wandb else "none"
        )
    
    def create_trainer(self, model, train_dataset, val_dataset, training_args) -> VQATrainer:
        """Create VQA trainer with curriculum learning support."""
        curriculum_scheduler = self.create_curriculum_scheduler()
        augmentation_factory = self.create_augmentation_factory()
        
        # Create sample observer if enabled
        sample_observer = None
        if self.config.training.enable_sample_observation:
            observation_dir = self.config.training.get_observation_dir(self.config.data.dataset_name)
            sample_observer = create_sample_observer(
                save_dir=str(observation_dir),
                num_samples=self.config.training.num_observation_samples
            )
            logger.info(f"Sample observation enabled: {observation_dir}")
        
        # Create wrong prediction tracker if enabled
        wrong_prediction_tracker = None
        if self.config.training.enable_wrong_prediction_tracking:
            wrong_pred_dir = self.config.training.get_wrong_prediction_dir(self.config.data.dataset_name)
            wrong_prediction_tracker = create_wrong_prediction_tracker(
                save_dir=str(wrong_pred_dir)
            )
            logger.info(f"Wrong prediction tracking enabled: {wrong_pred_dir}")
        
        if self.config.training.patience > 0:
            early_stopping = EarlyStoppingCallback(self.config.training.patience)
        else:
            early_stopping = None
        
        trainer = VQATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping] if early_stopping is not None else [],
            curriculum_scheduler=curriculum_scheduler,
            augmentation_factory=augmentation_factory,
            sample_observer=sample_observer,
            num_observation_samples=self.config.training.num_observation_samples,
            wrong_prediction_tracker=wrong_prediction_tracker,
        )
        
        return trainer
    
    def run(self):
        """Execute the complete training pipeline."""
        logger.info("=" * 80)
        logger.info("Starting VQA Training Pipeline")
        logger.info("=" * 80)
        
        # Setup WandB
        self.setup_wandb()
        
        # Create save directory
        save_dir = self.config.training.get_hf_save_dir(self.config.data.dataset_name)
        save_dir.parent.mkdir(exist_ok=True)
        logger.info(f"Save directory: {save_dir}")
        
        # Create processors
        text_processor, vis_processor = self.create_processors()
        
        # Create datasets
        train_dataset, val_dataset = self.create_datasets(text_processor, vis_processor)
        
        # Create model
        model, model_config = self.create_model(len(train_dataset.label_encoder))
        
        # Create training arguments
        training_args = self.create_training_arguments(save_dir)
        
        # Create trainer
        trainer = self.create_trainer(model, train_dataset, val_dataset, training_args)
        
        # Log additional info to WandB
        if self.config.training.report_to_wandb:
            wandb.config.update({
                "num_classes": len(train_dataset.label_encoder),
                "hidden_size": model_config.hidden_size,
            })
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("Training complete!")
        
        # Cleanup
        if self.config.training.report_to_wandb:
            wandb.finish()
        
        logger.info("=" * 80)
