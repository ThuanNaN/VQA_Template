"""
train.py - Training script với Replay Sampling per-Epoch

Chức năng chính:
- Fine-tune VQA model từ checkpoint có sẵn
- Sử dụng augmented dataset làm nguồn dữ liệu chính
- Replay sampling theo epoch: mỗi epoch mix augmented data + replay samples từ original/buffer
- Support resume training từ Hugging Face checkpoint

Cách chạy:
python train.py --checkpoint runs/checkpoint-3750 --replay_ratio 0.2 --replay_source original --epochs 10

Arguments chính:
--checkpoint: Đường dẫn tới checkpoint để resume training
--replay_ratio: Tỷ lệ replay samples trong mỗi epoch (0.0-1.0)
--replay_source: Nguồn replay (original/buffer)
--use_sap_combined: Sử dụng augmented dataset
"""

import os
from pathlib import Path
import wandb
import argparse
import torch
import random
import numpy as np
from dataset import ViVQADataset, OpenViVQADataset, ViVQAXDataset, CombinedDataset
from models import SimpleVQAConfig, SimpleVQA
from transformers import (
    AutoTokenizer, AutoProcessor, 
    TrainingArguments, EarlyStoppingCallback,
    set_seed, Trainer
)
import matplotlib.pyplot as plt
from utils import compute_metrics
import time
from torch.utils.data import ConcatDataset, Subset
import json

def set_global_seed(seed):
    """Set seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set environment variables for better reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # PyTorch deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable PyTorch deterministic algorithms (may slow down training)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Hugging Face transformers seed
    set_seed(seed)


def sample_indices(dataset, n_samples, seed):
    """
    Random sample indices từ dataset với seed cho reproducibility.
    
    Args:
        dataset: Dataset cần sample
        n_samples: Số lượng samples cần lấy
        seed: Random seed
        
    Returns:
        List of sampled indices
    """
    rng = np.random.RandomState(seed)
    dataset_size = len(dataset)
    
    if n_samples > dataset_size:
        print(f"⚠️  Warning: n_samples ({n_samples}) > dataset_size ({dataset_size})")
        print(f"   Sampling with replacement...")
        return rng.choice(dataset_size, size=n_samples, replace=True).tolist()
    
    return rng.choice(dataset_size, size=n_samples, replace=False).tolist()


def create_epoch_dataset(aug_dataset, replay_pool, replay_ratio, epoch, seed, log_indices=False):
    """
    Tạo dataset cho một epoch bằng cách mix augmented data + replay samples.
    
    Args:
        aug_dataset: Augmented dataset chính (hoặc combined dataset)
        replay_pool: Dataset nguồn để lấy replay samples (có thể là indices của original trong combined)
        replay_ratio: Tỷ lệ replay (0.0-1.0)
        epoch: Epoch hiện tại (để tạo seed khác nhau)
        seed: Base random seed
        log_indices: Log ra indices được sample
        
    Returns:
        Combined dataset cho epoch này
    """
    # Nếu replay_pool là list of indices (trường hợp CombinedDataset)
    if isinstance(replay_pool, list):
        # replay_pool là danh sách indices của original samples trong combined dataset
        # aug_dataset là danh sách indices của augmented samples trong combined dataset
        total_size = len(aug_dataset) + len(replay_pool)
        n_replay = int(replay_ratio * total_size)
        n_aug = total_size - n_replay
        
        print(f"\n📊 Building dataset for epoch {epoch}:")
        print(f"   Total size: {total_size}")
        print(f"   Augmented samples: {n_aug}")
        print(f"   Replay samples (from original): {n_replay}")
        
        # Sample từ augmented indices
        if n_aug > 0:
            rng = np.random.RandomState(seed + epoch * 1000)
            if n_aug > len(aug_dataset):
                aug_sampled_indices = rng.choice(aug_dataset, size=n_aug, replace=True).tolist()
            else:
                aug_sampled_indices = rng.choice(aug_dataset, size=n_aug, replace=False).tolist()
        else:
            aug_sampled_indices = []
        
        # Sample từ original indices (replay pool)
        if n_replay > 0:
            rng = np.random.RandomState(seed + epoch * 1000 + 1)
            if n_replay > len(replay_pool):
                replay_sampled_indices = rng.choice(replay_pool, size=n_replay, replace=True).tolist()
            else:
                replay_sampled_indices = rng.choice(replay_pool, size=n_replay, replace=False).tolist()
            
            if log_indices:
                print(f"   Replay indices (first 10): {replay_sampled_indices[:10]}")
        else:
            replay_sampled_indices = []
        
        # Combine indices
        combined_indices = aug_sampled_indices + replay_sampled_indices
        return combined_indices
    
    else:
        # Trường hợp cũ: replay_pool là một dataset riêng
        total_size = len(aug_dataset)
        n_replay = int(replay_ratio * total_size)
        n_aug = total_size - n_replay
        
        print(f"\n📊 Building dataset for epoch {epoch}:")
        print(f"   Total size: {total_size}")
        print(f"   Augmented samples: {n_aug}")
        print(f"   Replay samples: {n_replay}")
        
        # Sample từ augmented dataset (giảm bớt để dành chỗ cho replay)
        if n_aug > 0:
            aug_indices = sample_indices(aug_dataset, n_aug, seed=seed + epoch * 1000)
            aug_subset = Subset(aug_dataset, aug_indices)
        else:
            aug_subset = []
        
        # Sample replay từ replay pool
        if n_replay > 0:
            replay_indices = sample_indices(replay_pool, n_replay, seed=seed + epoch * 1000 + 1)
            replay_subset = Subset(replay_pool, replay_indices)
            
            if log_indices:
                print(f"   Replay indices (first 10): {replay_indices[:10]}")
        else:
            replay_subset = []
        
        # Concat datasets
        if n_aug > 0 and n_replay > 0:
            epoch_dataset = ConcatDataset([aug_subset, replay_subset])
        elif n_aug > 0:
            epoch_dataset = aug_subset
        else:
            epoch_dataset = replay_subset
        
        return epoch_dataset


def extract_original_samples(combined_dataset):
    """
    Trích xuất chỉ original samples từ CombinedDataset.
    
    Args:
        combined_dataset: CombinedDataset instance
        
    Returns:
        List of indices của original samples
    """
    original_indices = []
    for i, item in enumerate(combined_dataset.combined_data):
        if item.get('source') == 'original':
            original_indices.append(i)
    
    print(f"✅ Extracted {len(original_indices)} original samples from combined dataset")
    return original_indices


def extract_augmented_samples(combined_dataset):
    """
    Trích xuất chỉ augmented samples từ CombinedDataset.
    
    Args:
        combined_dataset: CombinedDataset instance
        
    Returns:
        List of indices của augmented samples
    """
    augmented_indices = []
    for i, item in enumerate(combined_dataset.combined_data):
        if item.get('source') == 'sap_augmented':
            augmented_indices.append(i)
    
    print(f"✅ Extracted {len(augmented_indices)} augmented samples from combined dataset")
    return augmented_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VQA model with replay sampling per-epoch')
    
    # Model & Dataset
    parser.add_argument('--vis_model_name', type=str, default='google/vit-base-patch16-224', 
                        choices=['google/vit-base-patch16-224', 'microsoft/beit-base-patch16-224-pt22k-ft22k'],
                        help='Vision model name (default: %(default)s)')
    parser.add_argument('--text_model_name', type=str, default='vinai/bartpho-syllable-base',
                        choices=['vinai/bartpho-syllable-base', 'vinai/bartpho-syllable', 'FacebookAI/xlm-roberta-base'],
                        help='Text model name (default: %(default)s)')
    parser.add_argument('--dataset_name', type=str, default='ViVQA', choices=['ViVQA', 'OpenViVQA', 'ViVQA-X'],
                        help='Dataset name (default: %(default)s)')
    parser.add_argument('--use_sap_combined', action='store_true',
                        help='Use CombinedDataset with SAP augmentation')
    
    # Replay Sampling Arguments
    parser.add_argument('--replay_ratio', type=float, default=0.0,
                        help='Tỷ lệ replay samples trong mỗi epoch (0.0-1.0, default: 0.0 = no replay). Requires --use_sap_combined')
    parser.add_argument('--log_replay_indices', action='store_true',
                        help='Log ra replay indices mỗi epoch để reproducibility')
    
    # Checkpoint & Resume
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Đường dẫn checkpoint để resume training')
    
    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=71,
                        help='random seed (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Mini-batch size for each iteration (default: %(default)s)')
    parser.add_argument('--seq_len', type=int, default=64,
                        help='Sequence length for text input (default: %(default)s)')
    parser.add_argument('--dataloader_workers', type=int, default=0,
                        help='Number of workers for dataloader (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train model (default: %(default)s)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer (default: %(default)s)')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Number of gradient accumulation steps (default: %(default)s)')
    parser.add_argument('--warmup_steps', type=int, default=250,
                        help='Number of warmup steps for learning rate scheduler (default: %(default)s)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Number of epochs to wait before early stopping (default: %(default)s)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    
    # Logging & Output
    parser.add_argument('--logging_steps', type=int, default=50,
                        help='Log training process every n steps (default: %(default)s)')
    parser.add_argument('--report_to_wandb', action='store_true',
                        help='Log training process to wandb')
    parser.add_argument('--wandb_name', type=str, default='VQA-Template',
                        help='Name of wandb project (default: %(default)s)')
    parser.add_argument('--output_dir', type=str, default='runs',
                        help='Output directory for model checkpoints (default: %(default)s)')
    parser.add_argument('--run_name', type=str, default='run',
                        help='Name of the run (default: %(default)s)')
    parser.add_argument('--n_threads', type=int, default=8,
                        help='Number of threads for torch (default: %(default)s)')
    
    args = parser.parse_args()
    
    # Validate replay arguments
    if args.replay_ratio < 0.0 or args.replay_ratio > 1.0:
        raise ValueError(f"replay_ratio must be between 0.0 and 1.0, got {args.replay_ratio}")
    
    if args.replay_ratio > 0 and not args.use_sap_combined:
        print("⚠️  Warning: replay_ratio > 0 but use_sap_combined=False")
        print("   Replay chỉ có ý nghĩa khi dùng augmented dataset")
    
    # Set global seed for reproducibility
    set_global_seed(args.seed)
    print(f"✅ Global seed set to: {args.seed}")
    
    # Setup directories and logging
    os.environ["WANDB_PROJECT"] = args.wandb_name
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    os.environ["WANDB_MODE"] = "offline"
    
    RUN_NAME = f"{args.dataset_name}-{args.run_name}-{args.seed}"
    if args.replay_ratio > 0:
        RUN_NAME += f"-replay{args.replay_ratio}"
    
    SAVE_DIR = Path(args.output_dir)
    SAVE_DIR.mkdir(exist_ok=True)
    HF_SAVE_DIR = SAVE_DIR / f"{RUN_NAME}_{int(time.time())}"
    
    # Setup threading
    system_threads = torch.get_num_threads()
    running_threads = args.n_threads
    if running_threads > system_threads:
        running_threads = system_threads
    torch.set_num_threads(running_threads)
    torch.set_num_interop_threads(running_threads)

    vis_model_name = args.vis_model_name
    text_model_name = args.text_model_name

    vis_processor = AutoProcessor.from_pretrained(vis_model_name, use_fast=True)
    text_processor = AutoTokenizer.from_pretrained(text_model_name)

    # =================================================================
    # LOAD DATASETS
    # =================================================================
    print("\n" + "="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    # Load training và validation datasets
    replay_pool = None
    augmented_indices = None
    
    if args.use_sap_combined:
        print(f"✅ Using augmented dataset (CombinedDataset)")
        # Load original datasets first
        if args.dataset_name == 'ViVQA':
            original_train_dataset = ViVQADataset(
                ann_path="data/vivqa/train.csv",
                img_dir="data/vivqa/images",
                text_processor=text_processor,
                vis_processor=vis_processor,
                max_length=args.seq_len
            )
            
            # Create combined dataset with augmented data
            train_dataset = CombinedDataset(
                original_dataset=original_train_dataset,
                augmented_json_path="simple_augmented_datasets/vivqa/ViVQA_simple_augmented.json",
            )
            
            # Validation dataset remains original
            val_dataset = ViVQADataset(
                ann_path="data/vivqa/test.csv",
                img_dir="data/vivqa/images",
                text_processor=text_processor,
                vis_processor=vis_processor,
                max_length=args.seq_len
            )

        elif args.dataset_name == 'OpenViVQA':
            original_train_dataset = OpenViVQADataset(
                ann_path="data/openvivqa/vlsp2023_train_data.json",
                img_dir="data/openvivqa/train-images",
                text_processor=text_processor,
                vis_processor=vis_processor,
                max_length=args.seq_len
            )
            
            train_dataset = CombinedDataset(
                original_dataset=original_train_dataset,
                augmented_ann_path="augmented_datasets/OpenViVQA_sap_augmented.json",
            )
            
            val_dataset = OpenViVQADataset(
                ann_path="data/openvivqa/vlsp2023_dev_data.json",
                img_dir="data/openvivqa/dev-images",
                text_processor=text_processor,
                vis_processor=vis_processor,
                max_length=args.seq_len
            )
        elif args.dataset_name == 'ViVQA-X':
            original_train_dataset = ViVQAXDataset(
                ann_path="data/vivqa-x/ViVQA-X_train.json",
                img_dir="data/MSCOCO/train2014",
                text_processor=text_processor,
                vis_processor=vis_processor,
                max_length=args.seq_len
            )
            
            train_dataset = CombinedDataset(
                original_dataset=original_train_dataset,
                augmented_json_path="augmented_datasets/ViVQA-X_sap_augmented.json",
            )

            val_dataset = ViVQAXDataset(
                ann_path="data/vivqa-x/ViVQA-X_val.json",
                img_dir="data/MSCOCO/val2014",
                text_processor=text_processor,
                vis_processor=vis_processor,
                max_length=args.seq_len
            )
        
        # Nếu có replay sampling, trích xuất original và augmented indices
        if args.replay_ratio > 0:
            print(f"\n🔄 Replay sampling enabled - extracting original/augmented samples")
            replay_pool = extract_original_samples(train_dataset)  # List of original indices
            augmented_indices = extract_augmented_samples(train_dataset)  # List of augmented indices
            print(f"   Will mix {len(augmented_indices)} augmented + replay from {len(replay_pool)} original per epoch")
        
    else:
        print(f"✅ Using original dataset (no augmentation)")
        # Original dataset loading logic
        if args.dataset_name == 'ViVQA':
            train_dataset = ViVQADataset(
                ann_path="data/vivqa/train.csv",
                img_dir="data/vivqa/images",
                text_processor=text_processor,
                vis_processor=vis_processor,
                max_length=args.seq_len
            )
            val_dataset = ViVQADataset(
                ann_path="data/vivqa/test.csv",
                img_dir="data/vivqa/images",
                text_processor=text_processor,
                vis_processor=vis_processor,
                max_length=args.seq_len
            )
        elif args.dataset_name == 'OpenViVQA':
            train_dataset = OpenViVQADataset(
                ann_path="data/openvivqa/vlsp2023_train_data.json",
                img_dir="data/openvivqa/training-images",
                text_processor=text_processor,
                vis_processor=vis_processor,
                max_length=args.seq_len
            )
            val_dataset = OpenViVQADataset(
                ann_path="data/openvivqa/vlsp2023_dev_data.json",
                img_dir="data/openvivqa/dev-images",
                text_processor=text_processor,
                vis_processor=vis_processor,
                max_length=args.seq_len
            )
        elif args.dataset_name == 'ViVQA-X':
            train_dataset = ViVQAXDataset(
                ann_path="data/vivqa-x/ViVQA-X_train.json",
                img_dir="data/MSCOCO/train2014",
                text_processor=text_processor,
                vis_processor=vis_processor,
                max_length=args.seq_len
            )
            val_dataset = ViVQAXDataset(
                ann_path="data/vivqa-x/ViVQA-X_val.json",
                img_dir="data/MSCOCO/val2014",
                text_processor=text_processor,
                vis_processor=vis_processor,
                max_length=args.seq_len
            )
        else:
            raise ValueError("Dataset name not found")
    
    print(f"✅ Train dataset size: {len(train_dataset)}")
    print(f"✅ Val dataset size: {len(val_dataset)}")
    if replay_pool is not None:
        print(f"✅ Replay pool size: {len(replay_pool)}")
    
    # =================================================================
    # CREATE MODEL
    # =================================================================
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    
    config = SimpleVQAConfig(
        vis_model_name=vis_model_name,
        text_model_name=text_model_name,
        num_classes=len(train_dataset.label_encoder)
    )
    model = SimpleVQA(config)
    print(f"✅ Model created with {len(train_dataset.label_encoder)} classes")
    
    # =================================================================
    # SETUP TRAINING ARGUMENTS
    # =================================================================
    save_safetensors = True
    if args.text_model_name in ['vinai/bartpho-syllable-base', 'vinai/bartpho-syllable']:
        save_safetensors = False

    training_args = TrainingArguments(
        seed=args.seed,
        data_seed=args.seed,
        output_dir=HF_SAVE_DIR,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        torch_compile=False,
        num_train_epochs=1 if args.replay_ratio > 0 else args.epochs,  # Per-epoch training nếu có replay
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False if args.replay_ratio > 0 else True,  # Disable vì train per-epoch
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch",
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        logging_dir="./logs",
        logging_steps=args.logging_steps,
        save_total_limit=3,
        push_to_hub=False,
        save_safetensors=save_safetensors,
        run_name=RUN_NAME,
        report_to="wandb" if args.report_to_wandb else "none",
    )

    early_stopping = EarlyStoppingCallback(args.patience)
    
    # =================================================================
    # TRAINING LOOP
    # =================================================================
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    if args.replay_ratio > 0:
        # Validate: replay chỉ hoạt động với use_sap_combined
        if not args.use_sap_combined:
            raise ValueError("Replay sampling requires --use_sap_combined to be enabled")
        
        # Training với replay sampling per-epoch
        print(f"🔄 Training mode: REPLAY SAMPLING")
        print(f"   Replay ratio: {args.replay_ratio}")
        print(f"   Total epochs: {args.epochs}")
        print(f"   Early stopping patience: {args.patience}")
        
        # Load checkpoint trước khi tạo trainer (nếu có)
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            
            # Nếu là file pytorch_model.bin, load trực tiếp vào model
            if checkpoint_path.is_file() and checkpoint_path.name == 'pytorch_model.bin':
                print(f"📂 Loading model weights from: {args.checkpoint}")
                state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
                model.load_state_dict(state_dict)
            # Nếu là directory, để Trainer.train() load
            elif checkpoint_path.is_dir():
                print(f"📂 Will resume from checkpoint directory: {args.checkpoint}")
            else:
                print(f"⚠️  Checkpoint not found or invalid: {args.checkpoint}")
        
        # Create trainer (sẽ update train_dataset mỗi epoch)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,  # Initial dataset (full combined)
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Early stopping tracking
        best_eval_loss = float('inf')
        best_epoch = -1
        patience_counter = 0
        best_checkpoint_dir = None
        
        # Training loop per-epoch
        for epoch in range(args.epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{args.epochs}")
            print(f"{'='*60}")
            
            # Build epoch indices (mix augmented + replay from original)
            epoch_indices = create_epoch_dataset(
                aug_dataset=augmented_indices,  # List of augmented indices
                replay_pool=replay_pool,  # List of original indices
                replay_ratio=args.replay_ratio,
                epoch=epoch,
                seed=args.seed,
                log_indices=args.log_replay_indices
            )
            
            # Create subset of combined dataset với indices đã sample
            epoch_dataset = Subset(train_dataset, epoch_indices)
            
            # Update trainer's dataset
            trainer.train_dataset = epoch_dataset
            
            # Train for this epoch
            set_global_seed(args.seed + epoch)  # Ensure reproducibility
            trainer.train(resume_from_checkpoint=False)
            
            # Evaluate
            eval_results = trainer.evaluate()
            current_eval_loss = eval_results['eval_loss']
            
            print(f"\n📊 Epoch {epoch + 1} Results:")
            print(f"   Eval Loss: {current_eval_loss:.4f}")
            if 'eval_accuracy' in eval_results:
                print(f"   Eval Accuracy: {eval_results['eval_accuracy']:.4f}")
            
            # Save checkpoint after each epoch
            epoch_checkpoint_dir = HF_SAVE_DIR / f"checkpoint-epoch-{epoch+1}"
            trainer.save_model(epoch_checkpoint_dir)
            print(f"💾 Saved checkpoint to: {epoch_checkpoint_dir}")
            
            # Early stopping check
            if current_eval_loss < best_eval_loss:
                best_eval_loss = current_eval_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_checkpoint_dir = epoch_checkpoint_dir
                print(f"✅ New best eval loss: {best_eval_loss:.4f}")
            else:
                patience_counter += 1
                print(f"⚠️  No improvement. Patience: {patience_counter}/{args.patience}")
                
                if patience_counter >= args.patience:
                    print(f"\n🛑 Early stopping triggered!")
                    print(f"   Best epoch: {best_epoch}")
                    print(f"   Best eval loss: {best_eval_loss:.4f}")
                    print(f"   Best checkpoint: {best_checkpoint_dir}")
                    break
        
        # Load best model if early stopping was used
        if best_checkpoint_dir and best_checkpoint_dir.exists():
            print(f"\n📂 Loading best model from: {best_checkpoint_dir}")
            trainer.model.load_state_dict(
                torch.load(best_checkpoint_dir / "pytorch_model.bin", map_location='cpu')
            )
    
    else:
        # Training thông thường (không replay)
        print(f"📚 Training mode: STANDARD")
        print(f"   Total epochs: {args.epochs}")
        
        # Load checkpoint trước nếu là file pytorch_model.bin
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            if checkpoint_path.is_file() and checkpoint_path.name == 'pytorch_model.bin':
                print(f"📂 Loading model weights from: {args.checkpoint}")
                state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
                model.load_state_dict(state_dict)
        
        set_global_seed(args.seed)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping],
        )
        
        # Train (chỉ resume từ directory checkpoint)
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            if checkpoint_path.is_dir():
                print(f"📂 Resuming from checkpoint directory: {args.checkpoint}")
                trainer.train(resume_from_checkpoint=args.checkpoint)
            else:
                # Đã load weights rồi, chỉ cần train
                trainer.train()
        else:
            trainer.train()
    
    # =================================================================
    # SAVE RESULTS & PLOTS
    # =================================================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    logs = trainer.state.log_history
    train_loss = [log["loss"] for log in logs if "loss" in log]
    eval_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
    eval_accuracy = [log["eval_accuracy"] for log in logs if "eval_accuracy" in log]
    epochs_range = range(1, len(eval_loss) + 1)

    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_loss)+1), train_loss, label="Train Loss")
    plt.plot(epochs_range, eval_loss, label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Evaluation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(HF_SAVE_DIR / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(epochs_range, eval_accuracy, label="Eval Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Evaluation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(HF_SAVE_DIR / "accuracy_curve.png")
    plt.close()
    
    print(f"✅ Training completed!")
    print(f"📁 Results saved to: {HF_SAVE_DIR}")
    wandb.finish()