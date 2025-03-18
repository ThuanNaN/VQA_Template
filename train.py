import wandb
import os
import argparse
from dataset import ViVQADataset, OpenViVQADataset
from transformers import AutoTokenizer, AutoProcessor
from models import SimpleVQAConfig, SimpleVQA
from transformers import TrainingArguments, Trainer
from utils import compute_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=59,
                        help='random seed will start at seed = 2 (default: %(default)s)')
    parser.add_argument('--dataset_name', type=str, default='ViVQA', choices=['ViVQA', 'OpenViVQA'],
                        help='Dataset name (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Mini-batch size for each iteration when training model (default: %(default)s)')
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
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Log training process every n steps (default: %(default)s)')
    parser.add_argument('--report_to_wandb', action='store_true',
                        help='Log training process to wandb')
    parser.add_argument('--wandb_name', type=str, default='VQA-Tempate',
                        help='Name of wandb project (default: %(default)s)')
    parser.add_argument('--output_dir', type=str, default='vqa_outputs',
                        help='Output directory for model checkpoints (default: %(default)s)')
    parser.add_argument('--run_name', type=str, default='run',
                        help='Name of the run (default: %(default)s)')
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"]=args.wandb_name
    os.environ["WANDB_LOG_MODEL"]="false"
    os.environ["WANDB_WATCH"]="false"

    vis_model_name = 'google/vit-base-patch16-224'
    text_model_name = 'vinai/phobert-base-v2'

    vis_processor = AutoProcessor.from_pretrained(vis_model_name, use_fast=True)
    text_processor = AutoTokenizer.from_pretrained(text_model_name)

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
            ann_path="data/openvivqa/train.csv",
            img_dir="data/openvivqa/images",
            text_processor=text_processor,
            vis_processor=vis_processor,
            max_length=args.seq_len
        )
        val_dataset = OpenViVQADataset(
            ann_path="data/openvivqa/test.csv",
            img_dir="data/openvivqa/images",
            text_processor=text_processor,
            vis_processor=vis_processor,
            max_length=args.seq_len
        )
    else:
        raise ValueError("Dataset name not found")
    
    num_classes = len(train_dataset.label_encoder)

    config = SimpleVQAConfig(
        vis_model_name=vis_model_name,
        text_model_name=text_model_name,
        hidden_size=768,
        num_classes=num_classes
    )
    model = SimpleVQA(config)

    training_args = TrainingArguments(
        seed=args.seed,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        torch_compile=True,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
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
        run_name=args.run_name,
        report_to="wandb" if args.report_to_wandb else "none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    wandb.finish()