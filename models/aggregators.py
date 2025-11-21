"""
Aggregation modules for combining multiple embeddings.
Supports both simple aggregation methods and learnable models.
"""

import torch
from torch import nn, Tensor
from typing import Literal, Union


class BaseAggregator(nn.Module):
    """Base class for all aggregators"""
    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Aggregate multiple embeddings into a single embedding.
        
        Args:
            embeddings: [batch_size, num_items, hidden_size]
        
        Returns:
            aggregated: [batch_size, hidden_size]
        """
        raise NotImplementedError


class SimpleAggregator(BaseAggregator):
    """Simple aggregation methods: mean, sum, max, first"""
    
    def __init__(self, method: Literal['mean', 'sum', 'max', 'first'] = 'mean'):
        super().__init__()
        self.method = method
    
    def forward(self, embeddings: Tensor) -> Tensor:
        if self.method == 'mean':
            return embeddings.mean(dim=1)
        elif self.method == 'sum':
            return embeddings.sum(dim=1)
        elif self.method == 'max':
            return embeddings.max(dim=1)[0]
        elif self.method == 'first':
            return embeddings[:, 0, :]
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")


class AttentionAggregator(BaseAggregator):
    """
    Learnable attention-based aggregation.
    Computes attention weights over multiple embeddings and returns weighted sum.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Learnable query vector for attention
        self.query = nn.Parameter(torch.randn(1, num_heads, hidden_size // num_heads))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
    
    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Args:
            embeddings: [batch_size, num_items, hidden_size]
        Returns:
            aggregated: [batch_size, hidden_size]
        """
        batch_size = embeddings.shape[0]
        
        # Expand query to batch size
        query = self.query.expand(batch_size, -1, -1)  # [batch, num_heads, hidden//num_heads]
        query = query.reshape(batch_size, 1, self.hidden_size)  # [batch, 1, hidden]
        
        # Apply multi-head attention
        # query: what we want, key/value: what we have
        aggregated, _ = self.attention(
            query=query,
            key=embeddings,
            value=embeddings
        )
        
        return aggregated.squeeze(1)  # [batch, hidden]


class TransformerAggregator(BaseAggregator):
    """
    Transformer-based aggregation.
    Processes embeddings through transformer layers then pools.
    """
    
    def __init__(self, hidden_size: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Add CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Args:
            embeddings: [batch_size, num_items, hidden_size]
        Returns:
            aggregated: [batch_size, hidden_size]
        """
        batch_size = embeddings.shape[0]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, hidden]
        embeddings_with_cls = torch.cat([cls_tokens, embeddings], dim=1)  # [batch, 1+num_items, hidden]
        
        # Process through transformer
        transformed = self.transformer(embeddings_with_cls)
        
        # Return CLS token representation
        return transformed[:, 0, :]  # [batch, hidden]


class GatedAggregator(BaseAggregator):
    """
    Gated aggregation with learnable weights.
    Each embedding gets a gate score, then weighted sum.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Args:
            embeddings: [batch_size, num_items, hidden_size]
        Returns:
            aggregated: [batch_size, hidden_size]
        """
        batch_size, num_items, hidden_size = embeddings.shape
        
        # Compute gate scores for each embedding
        gate_scores = self.gate(embeddings)  # [batch, num_items, 1]
        gate_weights = torch.softmax(gate_scores, dim=1)  # [batch, num_items, 1]
        
        # Weighted sum
        aggregated = (embeddings * gate_weights).sum(dim=1)  # [batch, hidden]
        
        return aggregated


class WeightedSumAggregator(BaseAggregator):
    """
    Simple learnable weighted sum with fixed weights per position.
    """
    
    def __init__(self, max_num_items: int = 10):
        super().__init__()
        self.max_num_items = max_num_items
        
        # Learnable weights for each position
        self.weights = nn.Parameter(torch.ones(max_num_items) / max_num_items)
    
    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Args:
            embeddings: [batch_size, num_items, hidden_size]
        Returns:
            aggregated: [batch_size, hidden_size]
        """
        num_items = embeddings.shape[1]
        
        # Get weights for available items and normalize
        weights = self.weights[:num_items]
        weights = torch.softmax(weights, dim=0)  # [num_items]
        
        # Weighted sum
        weights = weights.view(1, num_items, 1)  # [1, num_items, 1]
        aggregated = (embeddings * weights).sum(dim=1)  # [batch, hidden]
        
        return aggregated


def create_aggregator(
    aggregation: Union[str, nn.Module],
    hidden_size: int,
    **kwargs
) -> BaseAggregator:
    """
    Factory function to create aggregator from string or return existing module.
    
    Args:
        aggregation: Either:
            - String: 'mean', 'sum', 'max', 'first', 'attention', 'transformer', 'gated', 'weighted'
            - nn.Module: Custom aggregator instance
        hidden_size: Size of embeddings to aggregate
        **kwargs: Additional arguments for complex aggregators
            - num_heads: For attention aggregator (default: 1)
            - num_layers: For transformer aggregator (default: 2)
            - max_num_items: For weighted aggregator (default: 10)
    
    Returns:
        BaseAggregator instance
    """
    if isinstance(aggregation, nn.Module):
        # Already an aggregator module
        return aggregation
    
    if isinstance(aggregation, str):
        if aggregation in ['mean', 'sum', 'max', 'first']:
            return SimpleAggregator(method=aggregation)
        elif aggregation == 'attention':
            num_heads = kwargs.get('num_heads', 1)
            return AttentionAggregator(hidden_size, num_heads=num_heads)
        elif aggregation == 'transformer':
            num_layers = kwargs.get('num_layers', 2)
            num_heads = kwargs.get('num_heads', 4)
            dropout = kwargs.get('dropout', 0.1)
            return TransformerAggregator(hidden_size, num_layers=num_layers, 
                                        num_heads=num_heads, dropout=dropout)
        elif aggregation == 'gated':
            return GatedAggregator(hidden_size)
        elif aggregation == 'weighted':
            max_num_items = kwargs.get('max_num_items', 10)
            return WeightedSumAggregator(max_num_items=max_num_items)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    raise TypeError(f"Aggregation must be str or nn.Module, got {type(aggregation)}")
