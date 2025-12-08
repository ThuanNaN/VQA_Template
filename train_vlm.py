import os
import time
import json
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import wandb

from transformers import (
    set_seed,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# Import from project modules
from dataset.vlm_dataset import VLMVQADataset, VLMVQADatasetWithAugmentation
from models.vlm import VQACollator, evaluate_vqa_accuracy
from utils.vlm_utils import (
    get_label_encoder,
    get_dataset_paths,
    get_augmented_paths,
    set_global_seed,
    clear_memory
)
from utils.vqa_eval_callback import VQAEvalCallback

import warnings
warnings.filterwarnings("ignore")


def get_quantization_config(use_4bit: bool, use_8bit: bool = False):
    """Get quantization config for 4-bit or 8-bit loading."""
    if use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif use_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def get_lora_config(lora_r: int, lora_alpha: int, lora_dropout: float, target_modules: tuple):
    """Get LoRA configuration."""
    return LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        target_modules=list(target_modules),
        task_type="CAUSAL_LM",
    )


def load_vlm_model(
    model_name: str,
    use_4bit: bool = False,
    torch_dtype = torch.bfloat16,
    device_map: str = "auto",
    for_training: bool = True,
):
    """
    Load VLM model based on model name prefix.
    
    Args:
        model_name: Model name (e.g., "Qwen/Qwen2-VL-2B-Instruct", "google/paligemma-3b-pt-224")
        use_4bit: Use 4-bit quantization
        torch_dtype: Torch dtype for model
        device_map: Device map
        for_training: If True, disable cache for training
    
    Returns:
        Tuple of (model, processor)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name_lower = model_name.lower()
    
    # Detect model family by prefix
    if "qwen2-vl" in model_name_lower or "qwen/qwen2-vl" in model_name_lower:
        print(f"🔍 Detected: Qwen2-VL model")
        
        # Get quantization config
        quant_config = get_quantization_config(use_4bit) if device == "cuda" else None
        
        # Model kwargs
        model_kwargs = {
            "device_map": device_map if device == "cuda" else None,
            "use_cache": not for_training,
        }
        
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
            print(f"✅ Loading with 4-bit quantization")
        else:
            model_kwargs["torch_dtype"] = torch_dtype
            print(f"✅ Loading with dtype: {torch_dtype}")
        
        # Load model and processor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs
        )
        processor = Qwen2VLProcessor.from_pretrained(model_name)
        processor.tokenizer.padding_side = "right"
        
        print(f"✅ Model loaded: {model_name}")
        print(f"   Parameters: {model.num_parameters():,}")
        
        return model, processor
    
    elif "gemma" in model_name_lower or "paligemma" in model_name_lower:
        print(f"🔍 Detected: Gemma/PaliGemma model")
        raise NotImplementedError("Gemma/PaliGemma support coming soon!")
    
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported prefixes: qwen2-vl, gemma, paligemma")

def train_vlm(args):
    """Main training function for VLM."""
    
    # Setup
    set_global_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")
    print(f"✅ Seed: {args.seed}")
    
    # Setup directories and wandb (giống train.py)
    os.environ["WANDB_PROJECT"] = args.wandb_name
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    os.environ["WANDB_MODE"] = "offline"
    
    timestamp = int(time.time())
    run_name = f"{args.dataset_name}-VLM-{args.run_name}-{args.seed}_{timestamp}"
    save_dir = Path(args.output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Save directory: {save_dir}")
    
    # =================================================================
    # LOAD MODEL AND PROCESSOR
    # =================================================================
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    
    # Load model based on model name prefix
    model, processor = load_vlm_model(
        model_name=args.model_name,
        use_4bit=args.use_4bit,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        for_training=True
    )
    
    # Setup LoRA config (shared across all models)
    # Target modules có thể khác nhau tùy model, nhưng q_proj, v_proj thường universal
    lora_target_modules = ("q_proj", "v_proj")
    peft_config = get_lora_config(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules
    )
    
    print(f"✅ LoRA config:")
    print(f"   r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"   target_modules={lora_target_modules}")
    
    # =================================================================
    # LOAD DATASETS
    # =================================================================
    print("\n" + "="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    # Get label encoder and paths
    label_encoder = get_label_encoder(args.dataset_name)
    paths = get_dataset_paths(args.dataset_name)
    
    # Create datasets
    if args.use_sap_combined:
        print(f"✅ Using augmented dataset (SAP Combined)")
        augmented_path = get_augmented_paths(args.dataset_name)
        
        train_dataset = VLMVQADatasetWithAugmentation(
            ann_path=paths['train_ann'],
            img_dir=paths['train_img_dir'],
            label_encoder=label_encoder,
            augmented_json_path=augmented_path,
            max_augmented_samples=args.max_augmented_samples,
            split="train",
            language=args.language
        )
    else:
        print(f"✅ Using original dataset (no augmentation)")
        train_dataset = VLMVQADataset(
            ann_path=paths['train_ann'],
            img_dir=paths['train_img_dir'],
            label_encoder=label_encoder,
            split="train",
            language=args.language
        )
    
    eval_dataset = VLMVQADataset(
        ann_path=paths['val_ann'],
        img_dir=paths['val_img_dir'],
        label_encoder=label_encoder,
        split="val",
        language=args.language
    )
    
    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Eval dataset: {len(eval_dataset)} samples")
    print(f"✅ Number of classes: {len(label_encoder)}")
    
    # =================================================================
    # SETUP TRAINING ARGUMENTS
    # =================================================================
    print("\n" + "="*60)
    print("SETTING UP TRAINING")
    print("="*60)
    
    training_args = SFTConfig(
        output_dir=str(save_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",  # Evaluate per epoch
        save_strategy="epoch",  # Save per epoch
        save_total_limit=3,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        fp16=args.fp16 and device == "cuda",
        bf16=args.bf16 and device == "cuda",
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="wandb" if args.report_to_wandb else "none",
        run_name=run_name,
        seed=args.seed,
        data_seed=args.seed,
    )
    
    print(f"✅ Training config:")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Gradient accumulation: {args.gradient_accumulation}")
    print(f"   - Eval strategy: epoch")
    if args.use_sap_combined:
        print(f"   - Using augmented data: YES")
        if args.max_augmented_samples:
            print(f"   - Max augmented samples: {args.max_augmented_samples}")
    else:
        print(f"   - Using augmented data: NO")
    
    # =================================================================
    # CREATE TRAINER
    # =================================================================
    collate_fn = VQACollator(processor)
    
    # Create VQA eval callback (computes accuracy via generation)
    vqa_eval_callback = VQAEvalCallback(
        processor=processor,
        eval_dataset=eval_dataset,
        device=device,
        max_eval_samples=args.eval_max_samples,
        max_new_tokens=args.max_new_tokens
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor.tokenizer,
        callbacks=[vqa_eval_callback],
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Trainer created")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # =================================================================
    # INITIAL EVALUATION (optional)
    # =================================================================
    if args.eval_before_train:
        print("\n" + "="*60)
        print("INITIAL EVALUATION (Before Training)")
        print("="*60)
        
        # This will trigger vqa_eval_callback.on_evaluate() to compute accuracy
        initial_metrics = trainer.evaluate()
        print(f"Initial eval loss: {initial_metrics.get('eval_loss', 'N/A')}")
    
    # =================================================================
    # TRAINING
    # =================================================================
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    train_result = trainer.train()
    
    # Get final metrics from callback history
    final_accuracy = None
    if vqa_eval_callback.history:
        last_eval = vqa_eval_callback.history[-1]
        final_accuracy = last_eval['eval_accuracy']
        print(f"\n📊 Final Results from training:")
        print(f"   Best Eval Loss: {vqa_eval_callback.best_eval_loss:.4f}")
        print(f"   Best Accuracy: {vqa_eval_callback.best_accuracy:.4f}")
    
    # =================================================================
    # SAVE MODEL AND PLOTS
    # =================================================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save model
    trainer.save_model(str(save_dir / "final_model"))
    processor.save_pretrained(str(save_dir / "final_model"))
    print(f"✅ Model saved to: {save_dir / 'final_model'}")
    
    # Plot training curves from callback history
    logs = trainer.state.log_history
    train_losses = [log["loss"] for log in logs if "loss" in log and "eval_loss" not in log]
    
    # Get eval metrics from callback
    eval_epochs = [h['epoch'] for h in vqa_eval_callback.history]
    eval_losses = [h['eval_loss'] for h in vqa_eval_callback.history]
    eval_accuracies = [h['eval_accuracy'] for h in vqa_eval_callback.history]
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Train vs Eval Loss
    plt.subplot(1, 3, 1)
    if train_losses:
        plt.plot(train_losses, label="Train Loss", alpha=0.8)
    if eval_losses:
        eval_x = list(range(0, len(train_losses), max(1, len(train_losses) // len(eval_losses))))[:len(eval_losses)]
        plt.plot(eval_x, eval_losses, label="Eval Loss", marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training vs Evaluation Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Eval Loss per Epoch
    plt.subplot(1, 3, 2)
    if eval_losses:
        plt.plot(eval_epochs, eval_losses, label="Eval Loss", marker='o', color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Evaluation Loss per Epoch")
        plt.legend()
        plt.grid(True)
    
    # Plot 3: Accuracy per Epoch (giống train.py)
    plt.subplot(1, 3, 3)
    if eval_accuracies:
        plt.plot(eval_epochs, eval_accuracies, label="Eval Accuracy", marker='o', color='green')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Evaluation Accuracy per Epoch")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"✅ Training curves saved to: {save_dir / 'training_curves.png'}")
    
    # Save eval history
    with open(save_dir / "eval_history.json", 'w') as f:
        json.dump(vqa_eval_callback.history, f, indent=2)
    print(f"✅ Eval history saved to: {save_dir / 'eval_history.json'}")
    
    # Save training summary
    final_eval_loss = vqa_eval_callback.history[-1]['eval_loss'] if vqa_eval_callback.history else 0
    summary = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "use_augmented": args.use_sap_combined,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "num_classes": len(label_encoder),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "final_eval_loss": final_eval_loss,
        "final_accuracy": final_accuracy,
        "best_eval_loss": vqa_eval_callback.best_eval_loss,
        "best_accuracy": vqa_eval_callback.best_accuracy,
        "training_time": train_result.metrics.get('train_runtime', 0),
    }
    
    with open(save_dir / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Training summary saved to: {save_dir / 'training_summary.json'}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"📁 All results saved to: {save_dir}")
    wandb.finish()
    
    return trainer, model, processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune VLM for VQA task")
    
    # Model
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-2B-Instruct',
                        choices=['Qwen/Qwen2-VL-2B-Instruct', 'Qwen/Qwen2-VL-7B-Instruct'],
                        help='VLM model name')
    
    # Dataset
    parser.add_argument('--dataset_name', type=str, default='ViVQA',
                        choices=['ViVQA', 'OpenViVQA', 'ViVQA-X'],
                        help='Dataset name')
    parser.add_argument('--language', type=str, default='vi', choices=['vi', 'en'],
                        help='Language for system prompt')
    parser.add_argument('--use_sap_combined', action='store_true',
                        help='Use augmented dataset (SAP augmentation)')
    parser.add_argument('--max_augmented_samples', type=int, default=None,
                        help='Maximum number of augmented samples to use (None = use all). Requires --use_sap_combined')
    
    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--gradient_accumulation', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=50, help='Warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--max_new_tokens', type=int, default=32, help='Max new tokens for generation')
    
    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA r')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    
    # Quantization & Precision
    parser.add_argument('--use_4bit', action='store_true', default=True, help='Use 4-bit quantization (QLoRA)')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 training')
    parser.add_argument('--bf16', action='store_true', default=True, help='Use BF16 training')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True, help='Use gradient checkpointing')
    parser.add_argument('--optim', type=str, default='paged_adamw_32bit', help='Optimizer')
    
    # Evaluation
    parser.add_argument('--eval_before_train', action='store_true', help='Evaluate before training')
    parser.add_argument('--eval_max_samples', type=int, default=500, 
                        help='Max samples for accuracy evaluation per epoch (default: 500). Set to None for all samples')
    
    # Logging & Output
    parser.add_argument('--logging_steps', type=int, default=10, help='Logging steps')
    parser.add_argument('--dataloader_workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--output_dir', type=str, default='runs_vlm', help='Output directory')
    parser.add_argument('--run_name', type=str, default='run', help='Run name')
    parser.add_argument('--report_to_wandb', action='store_true', help='Report to wandb')
    parser.add_argument('--wandb_name', type=str, default='VQA-VLM-Finetune', help='Wandb project name')
    
    args = parser.parse_args()
    train_vlm(args)
