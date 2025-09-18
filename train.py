import os
from pathlib import Path
import wandb
import argparse
import torch
from dataset import ViVQADataset, OpenViVQADataset, ViVQAXDataset, CombinedDataset
from models import SimpleVQAConfig, SimpleVQA
from transformers import (
    AutoTokenizer, AutoProcessor, 
    TrainingArguments, EarlyStoppingCallback
)
import matplotlib.pyplot as plt
from utils import compute_metrics
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_model_name', type=str, default='google/vit-base-patch16-224', 
                        choices=['google/vit-base-patch16-224', 'microsoft/beit-base-patch16-224-pt22k-ft22k'],
                        help='Vision model name (default: %(default)s)')
    parser.add_argument('--text_model_name', type=str, default='vinai/bartpho-syllable-base',
                        choices=['vinai/bartpho-syllable-base', 'vinai/bartpho-syllable', 'FacebookAI/xlm-roberta-base'],
                        help='Text model name (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=71,
                        help='random seed (default: %(default)s)')
    parser.add_argument('--dataset_name', type=str, default='ViVQA-X', choices=['ViVQA', 'OpenViVQA', 'ViVQA-X'],
                        help='Dataset name (default: %(default)s)')
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
    parser.add_argument('--use_sap_combined', type=bool, default=False,
                        help='Use CombinedDataset that automatically includes SapAugmented data if available')
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"]=args.wandb_name
    os.environ["WANDB_LOG_MODEL"]="false"
    os.environ["WANDB_WATCH"]="false"
    RUN_NAME = f"{args.dataset_name}-{args.run_name}-{args.seed}"
    SAVE_DIR = Path(args.output_dir)
    SAVE_DIR.mkdir(exist_ok=True)
    HF_SAVE_DIR = SAVE_DIR / f"{RUN_NAME}_{int(time.time())}"

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

    # Use CombinedDataset if requested
    if args.use_sap_combined:
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
                augmented_ann_path="augmented_datasets/ViVQA_sap_augmented.json",
                text_processor=text_processor,
                vis_processor=vis_processor,
                max_length=args.seq_len
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
    else:
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

    config = SimpleVQAConfig(
        vis_model_name=vis_model_name,
        text_model_name=text_model_name,
        num_classes=len(train_dataset.label_encoder)
    )
    model = SimpleVQA(config)

    save_safetensors = True
    if args.text_model_name in ['vinai/bartpho-syllable-base', 'vinai/bartpho-syllable']:
        save_safetensors = False

    training_args = TrainingArguments(
        seed=args.seed,
        output_dir=HF_SAVE_DIR,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        torch_compile=True,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
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
        report_to="wandb" if args.report_to_wandb else "none"
    )

    early_stopping = EarlyStoppingCallback(args.patience)

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    trainer.train() 

    logs = trainer.state.log_history
    train_loss = [log["loss"] for log in logs if "loss" in log]
    eval_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
    eval_accuracy = [log["eval_accuracy"] for log in logs if "eval_accuracy" in log]
    epochs = range(1, len(eval_loss) + 1)

    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_loss)+1), train_loss, label="Train Loss")
    plt.plot(epochs, eval_loss, label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Evaluation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(HF_SAVE_DIR / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(epochs, eval_accuracy, label="Eval Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Evaluation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(HF_SAVE_DIR / "accuracy_curve.png")
    plt.close()
    wandb.finish()