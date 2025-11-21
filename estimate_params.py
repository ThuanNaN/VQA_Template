"""
Script to estimate model parameters for each component before training.
"""

import torch
from models.simple_vqa import SimpleVQA, SimpleVQAConfig
from training.config import ModelConfig
import argparse


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num):
    """Format large numbers in human-readable format."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def estimate_model_params(config: SimpleVQAConfig):
    """Estimate parameters for each part of the VQA model."""
    
    print("="*80)
    print("MODEL PARAMETER ESTIMATION")
    print("="*80)
    
    # Create model
    print("\nInitializing model...")
    model = SimpleVQA(config)
    
    # Overall model stats
    total_params, trainable_params = count_parameters(model)
    print(f"\n{'='*80}")
    print(f"OVERALL MODEL")
    print(f"{'='*80}")
    print(f"Total Parameters:      {format_number(total_params):>12} ({total_params:,})")
    print(f"Trainable Parameters:  {format_number(trainable_params):>12} ({trainable_params:,})")
    
    # Visual Encoder stats
    print(f"\n{'='*80}")
    print(f"VISUAL ENCODER")
    print(f"{'='*80}")
    print(f"Model: {config.vis_model_name}")
    
    vis_encoder_total, vis_encoder_trainable = count_parameters(model.vis_encoder.encoder)
    vis_proj_total, vis_proj_trainable = count_parameters(model.vis_encoder.proj)
    
    print(f"\nBackbone (Pretrained):")
    print(f"  Total:      {format_number(vis_encoder_total):>12} ({vis_encoder_total:,})")
    print(f"  Trainable:  {format_number(vis_encoder_trainable):>12} ({vis_encoder_trainable:,})")
    
    print(f"\nProjection Layer:")
    print(f"  Total:      {format_number(vis_proj_total):>12} ({vis_proj_total:,})")
    print(f"  Trainable:  {format_number(vis_proj_trainable):>12} ({vis_proj_trainable:,})")
    
    # Check if there's an aggregator with parameters
    vis_agg_total, vis_agg_trainable = count_parameters(model.vis_encoder.aggregator)
    if vis_agg_total > 0:
        print(f"\nAggregator ({config.vis_aggregation}):")
        print(f"  Total:      {format_number(vis_agg_total):>12} ({vis_agg_total:,})")
        print(f"  Trainable:  {format_number(vis_agg_trainable):>12} ({vis_agg_trainable:,})")
    else:
        print(f"\nAggregator: {config.vis_aggregation} (no parameters)")
    
    vis_total = vis_encoder_total + vis_proj_total + vis_agg_total
    print(f"\nVisual Encoder Total:  {format_number(vis_total):>12} ({vis_total:,})")
    
    # Text Encoder stats
    print(f"\n{'='*80}")
    print(f"TEXT ENCODER")
    print(f"{'='*80}")
    print(f"Model: {config.text_model_name}")
    
    text_encoder_total, text_encoder_trainable = count_parameters(model.text_encoder.encoder)
    text_proj_total, text_proj_trainable = count_parameters(model.text_encoder.proj)
    
    print(f"\nBackbone (Pretrained):")
    print(f"  Total:      {format_number(text_encoder_total):>12} ({text_encoder_total:,})")
    print(f"  Trainable:  {format_number(text_encoder_trainable):>12} ({text_encoder_trainable:,})")
    
    print(f"\nProjection Layer:")
    print(f"  Total:      {format_number(text_proj_total):>12} ({text_proj_total:,})")
    print(f"  Trainable:  {format_number(text_proj_trainable):>12} ({text_proj_trainable:,})")
    
    # Check if there's an aggregator with parameters
    text_agg_total, text_agg_trainable = count_parameters(model.text_encoder.aggregator)
    if text_agg_total > 0:
        print(f"\nAggregator ({config.text_aggregation}):")
        print(f"  Total:      {format_number(text_agg_total):>12} ({text_agg_total:,})")
        print(f"  Trainable:  {format_number(text_agg_trainable):>12} ({text_agg_trainable:,})")
    else:
        print(f"\nAggregator: {config.text_aggregation} (no parameters)")
    
    text_total = text_encoder_total + text_proj_total + text_agg_total
    print(f"\nText Encoder Total:    {format_number(text_total):>12} ({text_total:,})")
    
    # Classifier stats
    print(f"\n{'='*80}")
    print(f"CLASSIFIER")
    print(f"{'='*80}")
    classifier_total, classifier_trainable = count_parameters(model.classifier)
    
    fc_in_total, fc_in_trainable = count_parameters(model.classifier.fc_in)
    fc_out_total, fc_out_trainable = count_parameters(model.classifier.fc_out)
    
    print(f"Input Layer (2*{config.hidden_size} -> {config.hidden_size}):")
    print(f"  Total:      {format_number(fc_in_total):>12} ({fc_in_total:,})")
    print(f"  Trainable:  {format_number(fc_in_trainable):>12} ({fc_in_trainable:,})")
    
    print(f"\nOutput Layer ({config.hidden_size} -> {config.num_classes}):")
    print(f"  Total:      {format_number(fc_out_total):>12} ({fc_out_total:,})")
    print(f"  Trainable:  {format_number(fc_out_trainable):>12} ({fc_out_trainable:,})")
    
    print(f"\nClassifier Total:      {format_number(classifier_total):>12} ({classifier_total:,})")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY BY COMPONENT")
    print(f"{'='*80}")
    print(f"Visual Encoder:        {format_number(vis_total):>12} ({vis_total / total_params * 100:.2f}%)")
    print(f"Text Encoder:          {format_number(text_total):>12} ({text_total / total_params * 100:.2f}%)")
    print(f"Classifier:            {format_number(classifier_total):>12} ({classifier_total / total_params * 100:.2f}%)")
    print(f"{'-'*80}")
    print(f"Total:                 {format_number(total_params):>12} (100.00%)")
    
    print(f"\n{'='*80}")
    print(f"MEMORY ESTIMATION (FP32)")
    print(f"{'='*80}")
    # Each parameter is 4 bytes in FP32
    model_size_mb = (total_params * 4) / (1024 ** 2)
    print(f"Model Size:            {model_size_mb:>12.2f} MB")
    
    # Rule of thumb: training requires ~4x model size (model + gradients + optimizer states)
    training_mem_mb = model_size_mb * 4
    print(f"Est. Training Memory:  {training_mem_mb:>12.2f} MB (~{training_mem_mb/1024:.2f} GB)")
    
    print(f"{'='*80}\n")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Estimate VQA model parameters')
    parser.add_argument('--vis_model', type=str, default='google/vit-base-patch16-224',
                        help='Visual encoder model name')
    parser.add_argument('--text_model', type=str, default='vinai/bartpho-syllable-base',
                        help='Text encoder model name')
    parser.add_argument('--num_classes', type=int, default=500,
                        help='Number of output classes')
    parser.add_argument('--hidden_size', type=int, default=768,
                        help='Hidden size for projection')
    parser.add_argument('--text_aggregation', type=str, default='mean',
                        choices=['mean', 'sum', 'max', 'first', 'attention', 'transformer', 'gated', 'weighted'],
                        help='Text aggregation method')
    parser.add_argument('--vis_aggregation', type=str, default='mean',
                        choices=['mean', 'sum', 'max', 'first', 'attention', 'transformer', 'gated', 'weighted'],
                        help='Visual aggregation method')
    
    args = parser.parse_args()
    
    # Create config
    config = SimpleVQAConfig(
        vis_model_name=args.vis_model,
        text_model_name=args.text_model,
        num_classes=args.num_classes,
        hidden_size=args.hidden_size,
        text_aggregation=args.text_aggregation,
        vis_aggregation=args.vis_aggregation,
    )
    
    # Estimate parameters
    estimate_model_params(config)


if __name__ == "__main__":
    main()
