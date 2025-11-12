#!/bin/bash

# Example training script with wandb tracking
# This script demonstrates how to train a VQA model with wandb experiment tracking

# Make sure you have set WANDB_API_KEY in .env file or run 'wandb login' first

# Basic training with wandb
echo "Training ViVQA model with wandb tracking..."
python ../train.py \
    --dataset_name ViVQA \
    --vis_model_name google/vit-base-patch16-224 \
    --text_model_name vinai/bartpho-syllable-base \
    --batch_size 32 \
    --seq_len 64 \
    --epochs 30 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --gradient_accumulation 2 \
    --warmup_steps 250 \
    --patience 5 \
    --logging_steps 50 \
    --report_to_wandb \
    --wandb_name "VQA-Template" \
    --run_name "baseline-vit-bartpho" \
    --output_dir "../runs" \
    --seed 71

# You can also try different configurations:
# 
# # With mixed precision training
# python ../train.py \
#     --dataset_name ViVQA \
#     --batch_size 64 \
#     --epochs 30 \
#     --fp16 \
#     --report_to_wandb \
#     --wandb_name "VQA-Template" \
#     --run_name "fp16-training"
#
# # With OpenViVQA dataset
# python ../train.py \
#     --dataset_name OpenViVQA \
#     --batch_size 32 \
#     --epochs 30 \
#     --report_to_wandb \
#     --wandb_name "VQA-Template" \
#     --run_name "openvivqa-baseline"
#
# # With different models
# python ../train.py \
#     --dataset_name ViVQA \
#     --vis_model_name microsoft/beit-base-patch16-224-pt22k-ft22k \
#     --text_model_name FacebookAI/xlm-roberta-base \
#     --report_to_wandb \
#     --wandb_name "VQA-Template" \
#     --run_name "beit-xlmroberta"
