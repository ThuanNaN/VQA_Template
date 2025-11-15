"""
Example usage of different aggregation methods for VQA model.
Demonstrates both simple and complex aggregators.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models import (
    SimpleVQA, 
    SimpleVQAConfig,
    AttentionAggregator,
    TransformerAggregator,
    GatedAggregator
)


def example_simple_aggregations():
    """Example using simple string-based aggregations"""
    print("=" * 60)
    print("Example 1: Simple String-Based Aggregations")
    print("=" * 60)
    
    aggregation_methods = ['mean', 'sum', 'max', 'first']
    
    for method in aggregation_methods:
        print(f"\n--- Using '{method}' aggregation ---")
        
        config = SimpleVQAConfig(
            vis_model_name='google/vit-base-patch16-224',
            text_model_name='vinai/bartpho-syllable-base',
            num_classes=1000,
            hidden_size=512,
            text_aggregation=method,  # Simple string method
            vis_aggregation=method
        )
        
        model = SimpleVQA(config)
        print(f"Text aggregator type: {type(model.text_encoder.aggregator).__name__}")
        print(f"Vis aggregator type: {type(model.vis_encoder.aggregator).__name__}")


def example_attention_aggregation():
    """Example using learnable attention-based aggregation"""
    print("\n" + "=" * 60)
    print("Example 2: Attention-Based Aggregation")
    print("=" * 60)
    
    config = SimpleVQAConfig(
        vis_model_name='google/vit-base-patch16-224',
        text_model_name='vinai/bartpho-syllable-base',
        num_classes=1000,
        hidden_size=512,
        text_aggregation='attention',  # Attention aggregator
        vis_aggregation='attention',
        text_aggregation_kwargs={'num_heads': 4},  # 4-head attention
        vis_aggregation_kwargs={'num_heads': 8}    # 8-head attention
    )
    
    model = SimpleVQA(config)
    
    print(f"\nText aggregator: {type(model.text_encoder.aggregator).__name__}")
    print(f"  - Attention heads: {model.text_encoder.aggregator.num_heads}")
    print(f"\nVis aggregator: {type(model.vis_encoder.aggregator).__name__}")
    print(f"  - Attention heads: {model.vis_encoder.aggregator.num_heads}")
    
    # Test with dummy data
    batch_size = 2
    num_paraphrases = 3
    
    # Simulate multiple paraphrases
    dummy_input_ids = torch.randint(0, 1000, (batch_size, num_paraphrases, 128))
    dummy_attention_mask = torch.ones(batch_size, num_paraphrases, 128)
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_labels = torch.randint(0, 1000, (batch_size,))
    
    print(f"\nInput shapes:")
    print(f"  - Question input_ids: {dummy_input_ids.shape}")
    print(f"  - Images: {dummy_images.shape}")
    
    output = model(
        image=dummy_images,
        question_input_ids=dummy_input_ids,
        question_attention_mask=dummy_attention_mask,
        labels=dummy_labels
    )
    
    print(f"\nOutput:")
    print(f"  - Logits shape: {output['logits'].shape}")
    print(f"  - Loss: {output['loss'].item():.4f}")


def example_transformer_aggregation():
    """Example using transformer-based aggregation"""
    print("\n" + "=" * 60)
    print("Example 3: Transformer-Based Aggregation")
    print("=" * 60)
    
    config = SimpleVQAConfig(
        vis_model_name='google/vit-base-patch16-224',
        text_model_name='vinai/bartpho-syllable-base',
        num_classes=1000,
        hidden_size=512,
        text_aggregation='transformer',
        vis_aggregation='mean',  # Mix different aggregators
        text_aggregation_kwargs={
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1
        }
    )
    
    model = SimpleVQA(config)
    
    print(f"\nText aggregator: {type(model.text_encoder.aggregator).__name__}")
    print(f"  - Transformer layers: {model.text_encoder.aggregator.transformer.num_layers}")
    print(f"  - Attention heads: 4")
    print(f"\nVis aggregator: {type(model.vis_encoder.aggregator).__name__}")
    
    # Count parameters
    text_agg_params = sum(p.numel() for p in model.text_encoder.aggregator.parameters())
    print(f"\nText aggregator parameters: {text_agg_params:,}")


def example_gated_aggregation():
    """Example using gated aggregation"""
    print("\n" + "=" * 60)
    print("Example 4: Gated Aggregation")
    print("=" * 60)
    
    config = SimpleVQAConfig(
        vis_model_name='google/vit-base-patch16-224',
        text_model_name='vinai/bartpho-syllable-base',
        num_classes=1000,
        hidden_size=512,
        text_aggregation='gated',
        vis_aggregation='gated'
    )
    
    model = SimpleVQA(config)
    
    print(f"\nText aggregator: {type(model.text_encoder.aggregator).__name__}")
    print(f"Vis aggregator: {type(model.vis_encoder.aggregator).__name__}")
    
    # Count parameters
    text_agg_params = sum(p.numel() for p in model.text_encoder.aggregator.parameters())
    vis_agg_params = sum(p.numel() for p in model.vis_encoder.aggregator.parameters())
    
    print(f"\nText aggregator parameters: {text_agg_params:,}")
    print(f"Vis aggregator parameters: {vis_agg_params:,}")


def example_custom_aggregator():
    """Example using custom aggregator module"""
    print("\n" + "=" * 60)
    print("Example 5: Custom Aggregator Module")
    print("=" * 60)
    
    # Create custom aggregator instances
    custom_text_agg = AttentionAggregator(hidden_size=512, num_heads=8)
    custom_vis_agg = TransformerAggregator(hidden_size=512, num_layers=3, num_heads=8)
    
    config = SimpleVQAConfig(
        vis_model_name='google/vit-base-patch16-224',
        text_model_name='vinai/bartpho-syllable-base',
        num_classes=1000,
        hidden_size=512,
        text_aggregation=custom_text_agg,  # Pass nn.Module directly
        vis_aggregation=custom_vis_agg
    )
    
    model = SimpleVQA(config)
    
    print(f"\nText aggregator: {type(model.text_encoder.aggregator).__name__}")
    print(f"  - Is same instance: {model.text_encoder.aggregator is custom_text_agg}")
    print(f"\nVis aggregator: {type(model.vis_encoder.aggregator).__name__}")
    print(f"  - Is same instance: {model.vis_encoder.aggregator is custom_vis_agg}")


def example_mixed_aggregations():
    """Example mixing different aggregation types"""
    print("\n" + "=" * 60)
    print("Example 6: Mixed Aggregation Strategies")
    print("=" * 60)
    
    configs = [
        {
            'name': 'Simple text + Complex visual',
            'text': 'mean',
            'vis': 'transformer',
            'vis_kwargs': {'num_layers': 2, 'num_heads': 4}
        },
        {
            'name': 'Complex text + Simple visual',
            'text': 'attention',
            'vis': 'max',
            'text_kwargs': {'num_heads': 8}
        },
        {
            'name': 'Both complex',
            'text': 'gated',
            'vis': 'attention',
            'vis_kwargs': {'num_heads': 4}
        }
    ]
    
    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        
        config = SimpleVQAConfig(
            vis_model_name='google/vit-base-patch16-224',
            text_model_name='vinai/bartpho-syllable-base',
            num_classes=1000,
            hidden_size=512,
            text_aggregation=cfg['text'],
            vis_aggregation=cfg['vis'],
            text_aggregation_kwargs=cfg.get('text_kwargs', {}),
            vis_aggregation_kwargs=cfg.get('vis_kwargs', {})
        )
        
        model = SimpleVQA(config)
        
        text_params = sum(p.numel() for p in model.text_encoder.aggregator.parameters())
        vis_params = sum(p.numel() for p in model.vis_encoder.aggregator.parameters())
        
        print(f"  Text: {type(model.text_encoder.aggregator).__name__} ({text_params:,} params)")
        print(f"  Vis:  {type(model.vis_encoder.aggregator).__name__} ({vis_params:,} params)")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("VQA Aggregation Methods Examples")
    print("="*60)
    
    # Run all examples
    example_simple_aggregations()
    example_attention_aggregation()
    example_transformer_aggregation()
    example_gated_aggregation()
    example_custom_aggregator()
    example_mixed_aggregations()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
