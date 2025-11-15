"""
Training configuration management.

This module provides dataclasses for managing training configurations
in a type-safe and organized manner.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Union, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    vis_model_name: str = 'google/vit-base-patch16-224'
    text_model_name: str = 'vinai/bartpho-syllable-base'
    num_classes: Optional[int] = None
    hidden_size: int = 768
    dropout: float = 0.1
    
    # Aggregation strategies for multiple inputs
    # Simple: 'mean', 'sum', 'max', 'first'
    # Complex: 'attention', 'transformer', 'gated', 'weighted'
    text_aggregation: str = 'mean'
    vis_aggregation: str = 'mean'
    
    # Additional kwargs for complex aggregators
    # Example for attention: {'num_heads': 4, 'dropout': 0.1}
    # Example for transformer: {'num_layers': 2, 'num_heads': 4}
    text_aggregation_kwargs: dict = field(default_factory=dict)
    vis_aggregation_kwargs: dict = field(default_factory=dict)


@dataclass
class DataConfig:
    """Configuration for dataset."""
    dataset_name: str = 'vivqa'
    train_ann_path: Optional[str] = None
    val_ann_path: Optional[str] = None
    train_img_dir: Optional[str] = None
    val_img_dir: Optional[str] = None
    seq_len: int = 64
    batch_size: int = 64
    dataloader_workers: int = 0
    
    def __post_init__(self):
        """Auto-populate paths based on dataset name if not provided."""
        if self.train_ann_path is None or self.val_ann_path is None:
            self._set_default_paths()
    
    def _set_default_paths(self):
        """Set default paths based on dataset name."""
        dataset_map = {
            'vivqa': {
                'train_ann': 'data/vivqa/train.csv',
                'val_ann': 'data/vivqa/test.csv',
                'train_img': 'data/vivqa/images',
                'val_img': 'data/vivqa/images',
            },
            'openvivqa': {
                'train_ann': 'data/openvivqa/vlsp2023_train_data.json',
                'val_ann': 'data/openvivqa/vlsp2023_dev_data.json',
                'train_img': 'data/openvivqa/training-images',
                'val_img': 'data/openvivqa/dev-images',
            },
            'vitextvqa': {
                'train_ann': 'data/vitextvqa/ViTextVQA_train.json',
                'val_ann': 'data/vitextvqa/ViTextVQA_dev.json',
                'train_img': 'data/vitextvqa/images',
                'val_img': 'data/vitextvqa/images',
            },
            'evjvqa': {
                'train_ann': 'data/evjvqa/train.json',
                'val_ann': 'data/evjvqa/val.json',
                'train_img': 'data/evjvqa/images',
                'val_img': 'data/evjvqa/images',
            },
            'vivqax': {
                'train_ann': 'data/vivqax/ViVQA-X_train.json',
                'val_ann': 'data/vivqax/ViVQA-X_val.json',
                'train_img': 'data/vivqax/images',
                'val_img': 'data/vivqax/images',
            },
            'viocrvqa': {
                'train_ann': 'data/viocrvqa/train.json',
                'val_ann': 'data/viocrvqa/dev.json',
                'train_img': 'data/viocrvqa/images',
                'val_img': 'data/viocrvqa/images',
            },
        }
        
        if self.dataset_name in dataset_map:
            paths = dataset_map[self.dataset_name]
            if self.train_ann_path is None:
                self.train_ann_path = paths['train_ann']
            if self.val_ann_path is None:
                self.val_ann_path = paths['val_ann']
            if self.train_img_dir is None:
                self.train_img_dir = paths['train_img']
            if self.val_img_dir is None:
                self.val_img_dir = paths['val_img']


@dataclass
class AugmentationConfig:
    """Configuration for augmentation."""
    # Image augmentation
    enable_image_augmentation: bool = False
    image_augmentation_type: str = 'masked'
    patch_size: int = 16
    
    # Text augmentation
    enable_text_augmentation: bool = False
    text_augmentation_type: str = 'simple'
    
    # General
    seed: Optional[int] = None
    
    # Curriculum learning settings
    enable_curriculum: bool = False
    total_epochs: Optional[int] = None
    easy_epochs: Optional[int] = None
    medium_epochs: Optional[int] = None
    hard_epochs: Optional[int] = None
    
    @property
    def enable_augmentation(self) -> bool:
        """Check if any augmentation is enabled."""
        return self.enable_image_augmentation or self.enable_text_augmentation


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    # Basic settings
    seed: int = 71
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Training optimization
    gradient_accumulation: int = 1
    warmup_steps: int = 250
    fp16: bool = False
    
    # Early stopping
    patience: int = 3
    
    # Logging and saving
    logging_steps: int = 50
    output_dir: str = 'runs'
    run_name: str = 'run'
    
    # Sample observation
    enable_sample_observation: bool = True
    observation_dir: str = 'observations'
    num_observation_samples: int = 5
    
    # Wrong prediction tracking
    enable_wrong_prediction_tracking: bool = True
    wrong_prediction_dir: str = 'wrong_predictions'
    
    # Weights & Biases
    report_to_wandb: bool = False
    wandb_project: str = 'VQA-Template'
    
    # System
    n_threads: int = 8
    
    def get_hf_save_dir(self, dataset_name: str) -> Path:
        """Get HuggingFace save directory."""
        run_name = f"{dataset_name}-{self.run_name}-{self.seed}"
        return Path(self.output_dir) / run_name
    
    def get_observation_dir(self, dataset_name: str) -> Path:
        """Get observation save directory."""
        run_name = f"{dataset_name}-{self.run_name}-{self.seed}"
        return Path(self.observation_dir) / run_name
    
    def get_wrong_prediction_dir(self, dataset_name: str) -> Path:
        """Get wrong prediction save directory."""
        run_name = f"{dataset_name}-{self.run_name}-{self.seed}"
        return Path(self.wrong_prediction_dir) / run_name


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'augmentation': asdict(self.augmentation),
            'training': asdict(self.training),
        }
    
    @classmethod
    def from_args(cls, args):
        """Create configuration from argparse arguments."""
        model_config = ModelConfig(
            vis_model_name=args.vis_model_name,
            text_model_name=args.text_model_name,
            hidden_size=getattr(args, 'hidden_size', 768),
            text_aggregation=getattr(args, 'text_aggregation', 'mean'),
            vis_aggregation=getattr(args, 'vis_aggregation', 'mean'),
            text_aggregation_kwargs=getattr(args, 'text_aggregation_kwargs', {}),
            vis_aggregation_kwargs=getattr(args, 'vis_aggregation_kwargs', {}),
        )
        
        data_config = DataConfig(
            dataset_name=args.dataset_name,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            dataloader_workers=args.dataloader_workers,
        )
        
        augmentation_config = AugmentationConfig(
            enable_image_augmentation=getattr(args, 'enable_image_augmentation', False),
            image_augmentation_type=getattr(args, 'image_augmentation_type', 'masked'),
            patch_size=getattr(args, 'patch_size', 16),
            enable_text_augmentation=getattr(args, 'enable_text_augmentation', False),
            text_augmentation_type=getattr(args, 'text_augmentation_type', 'simple'),
            seed=args.seed,
            enable_curriculum=getattr(args, 'enable_curriculum', False),
            total_epochs=args.epochs,
            easy_epochs=getattr(args, 'easy_epochs', None),
            medium_epochs=getattr(args, 'medium_epochs', None),
            hard_epochs=getattr(args, 'hard_epochs', None),
        )
        
        training_config = TrainingConfig(
            seed=args.seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_accumulation=args.gradient_accumulation,
            warmup_steps=args.warmup_steps,
            fp16=args.fp16,
            patience=args.patience,
            logging_steps=args.logging_steps,
            output_dir=args.output_dir,
            run_name=args.run_name,
            enable_sample_observation=getattr(args, 'enable_sample_observation', False),
            observation_dir=getattr(args, 'observation_dir', 'observations'),
            num_observation_samples=getattr(args, 'num_observation_samples', 5),
            enable_wrong_prediction_tracking=getattr(args, 'enable_wrong_prediction_tracking', True),
            wrong_prediction_dir=getattr(args, 'wrong_prediction_dir', 'wrong_predictions'),
            report_to_wandb=args.report_to_wandb,
            wandb_project=getattr(args, 'wandb_name', 'VQA-Template'),
            n_threads=args.n_threads,
        )
        
        return cls(
            model=model_config,
            data=data_config,
            augmentation=augmentation_config,
            training=training_config,
        )
