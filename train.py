"""
Training script for VQA models with OOP architecture.

This script demonstrates the new OOP-based training pipeline with:
- Structured configuration management
- Curriculum learning support
- Dynamic augmentation
- Clean separation of concerns
"""

import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from training import ExperimentConfig, VQATrainingPipeline

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train VQA models with curriculum learning and augmentation support'
    )
    
    # Model arguments
    parser.add_argument(
        '--vis_model_name',
        type=str,
        default='google/vit-base-patch16-224',
        choices=['google/vit-base-patch16-224', 'microsoft/beit-base-patch16-224-pt22k-ft22k'],
        help='Vision model name (default: %(default)s)'
    )
    parser.add_argument(
        '--text_model_name',
        type=str,
        default='vinai/bartpho-syllable-base',
        choices=['vinai/bartpho-syllable-base', 'vinai/bartpho-syllable', 'FacebookAI/xlm-roberta-base'],
        help='Text model name (default: %(default)s)'
    )
    
    # Dataset arguments
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='vivqa',
        choices=['vivqa', 'openvivqa', 'vitextvqa', 'evjvqa', 'vivqax', 'viocrvqa'],
        help='Dataset name (default: %(default)s)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Mini-batch size for each iteration (default: %(default)s)'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=64,
        help='Sequence length for text input (default: %(default)s)'
    )
    parser.add_argument(
        '--dataloader_workers',
        type=int,
        default=0,
        help='Number of workers for dataloader (default: %(default)s)'
    )
    
    # Training arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=71,
        help='Random seed (default: %(default)s)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of epochs to train model (default: %(default)s)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Initial learning rate (default: %(default)s)'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay for optimizer (default: %(default)s)'
    )
    parser.add_argument(
        '--gradient_accumulation',
        type=int,
        default=1,
        help='Number of gradient accumulation steps (default: %(default)s)'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=250,
        help='Number of warmup steps for learning rate scheduler (default: %(default)s)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='Number of epochs to wait before early stopping (default: %(default)s)'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use mixed precision training'
    )
    
    # Augmentation arguments
    parser.add_argument(
        '--enable_image_augmentation',
        action='store_true',
        help='Enable image augmentation'
    )
    parser.add_argument(
        '--image_augmentation_type',
        type=str,
        default='masked',
        choices=['masked', 'mae', 'none'],
        help='Type of image augmentation (default: %(default)s)'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=16,
        help='Patch size for masked augmentation (default: %(default)s)'
    )
    parser.add_argument(
        '--enable_text_augmentation',
        action='store_true',
        help='Enable text augmentation'
    )
    parser.add_argument(
        '--text_augmentation_type',
        type=str,
        default='simple',
        choices=['simple', 'rule_based', 'none'],
        help='Type of text augmentation (default: %(default)s)'
    )
    
    # Curriculum learning arguments
    parser.add_argument(
        '--enable_curriculum',
        action='store_true',
        help='Enable curriculum learning'
    )
    parser.add_argument(
        '--easy_epochs',
        type=int,
        default=None,
        help='Number of epochs for easy difficulty (default: 30%% of total)'
    )
    parser.add_argument(
        '--medium_epochs',
        type=int,
        default=None,
        help='Number of epochs for medium difficulty (default: 30%% of total)'
    )
    parser.add_argument(
        '--hard_epochs',
        type=int,
        default=None,
        help='Number of epochs for hard difficulty (default: remaining)'
    )
    
    # Logging and saving arguments
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=50,
        help='Log training process every n steps (default: %(default)s)'
    )
    parser.add_argument(
        '--report_to_wandb',
        action='store_true',
        help='Log training process to wandb'
    )
    parser.add_argument(
        '--wandb_name',
        type=str,
        default='VQA-Template',
        help='Name of wandb project (default: %(default)s)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='runs',
        help='Output directory for model checkpoints (default: %(default)s)'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default='run',
        help='Name of the run (default: %(default)s)'
    )
    
    # System arguments
    parser.add_argument(
        '--n_threads',
        type=int,
        default=8,
        help='Number of threads for torch (default: %(default)s)'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Create experiment configuration
    config = ExperimentConfig.from_args(args)
    
    logger.info("Experiment Configuration:")
    logger.info(f"  Model: {config.model.vis_model_name} + {config.model.text_model_name}")
    logger.info(f"  Dataset: {config.data.dataset_name}")
    logger.info(f"  Image Augmentation: {config.augmentation.enable_image_augmentation} ({config.augmentation.image_augmentation_type})")
    logger.info(f"  Text Augmentation: {config.augmentation.enable_text_augmentation} ({config.augmentation.text_augmentation_type})")
    logger.info(f"  Curriculum: {config.augmentation.enable_curriculum}")
    logger.info(f"  Epochs: {config.training.epochs}")
    logger.info(f"  Batch size: {config.data.batch_size}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    
    # Create and run pipeline
    pipeline = VQATrainingPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()
