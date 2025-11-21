from typing import Optional, Union
import torch
from torch import nn, Tensor
from .base import (
    BaseTextEncoder,
    BaseVisEncoder,
    BaseClassifier,
    BaseVQA
)
from .aggregators import create_aggregator, BaseAggregator
from transformers import AutoModel
from dataclasses import dataclass, field

@dataclass
class SimpleVQAConfig:
    vis_model_name: str
    text_model_name: str
    num_classes: int
    hidden_size: int = 1024
    # Can be: 'mean', 'sum', 'max', 'first', 'attention', 'transformer', 'gated', 'weighted', or nn.Module
    text_aggregation: Union[str, nn.Module] = 'mean'
    vis_aggregation: Union[str, nn.Module] = 'mean'
    # Additional kwargs for complex aggregators
    text_aggregation_kwargs: dict = field(default_factory=dict)
    vis_aggregation_kwargs: dict = field(default_factory=dict)


class TextEncoder(BaseTextEncoder):
    def __init__(self, model_name: str, hidden_size: int, 
                 aggregation: Union[str, BaseAggregator] = 'mean',
                 aggregation_kwargs: dict = None):
        super(TextEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.encoder.config.hidden_size, hidden_size)
        
        # Create aggregator (can be simple string or complex module)
        aggregation_kwargs = aggregation_kwargs or {}
        self.aggregator = create_aggregator(aggregation, hidden_size, **aggregation_kwargs)

    def forward(self, inputs: dict) -> Tensor:
        """
        Forward pass supporting both single and multiple text inputs.
        
        Args:
            inputs: Dict with 'input_ids' and 'attention_mask'
                   - Single: [batch_size, seq_len]
                   - Multiple: [batch_size, num_texts, seq_len]
        
        Returns:
            Text features: [batch_size, hidden_size]
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Check if we have multiple texts per sample
        if input_ids.dim() == 3:
            # Multiple texts: [batch_size, num_texts, seq_len]
            batch_size, num_texts, seq_len = input_ids.shape
            
            # Flatten to process all texts together
            input_ids_flat = input_ids.view(batch_size * num_texts, seq_len)
            attention_mask_flat = attention_mask.view(batch_size * num_texts, seq_len)
            
            # Encode all texts
            outputs = self.encoder(
                input_ids=input_ids_flat,
                attention_mask=attention_mask_flat
            )
            embeddings = outputs.last_hidden_state[:, 0, :]  # [batch*num_texts, hidden]
            
            # Reshape back to [batch_size, num_texts, hidden]
            embeddings = embeddings.view(batch_size, num_texts, -1)
            
            # Project first, then aggregate (ensures correct hidden_size for aggregator)
            projected = self.proj(embeddings.view(batch_size * num_texts, -1))
            projected = projected.view(batch_size, num_texts, -1)
            
            # Aggregate embeddings using the aggregator
            return self.aggregator(projected)
        else:
            # Single text: [batch_size, seq_len]
            outputs = self.encoder(**inputs)
            last_hidden_state = outputs.last_hidden_state[:, 0, :]
            return self.proj(last_hidden_state)


class VisEncoder(BaseVisEncoder):
    def __init__(self, model_name: str, hidden_size: int, 
                 aggregation: Union[str, BaseAggregator] = 'mean',
                 aggregation_kwargs: dict = None):
        super(VisEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.encoder.config.hidden_size, hidden_size)
        
        # Create aggregator (can be simple string or complex module)
        aggregation_kwargs = aggregation_kwargs or {}
        self.aggregator = create_aggregator(aggregation, hidden_size, **aggregation_kwargs)

    def forward(self, inputs: dict) -> Tensor:
        """
        Forward pass supporting both single and multiple image inputs.
        
        Args:
            inputs: Dict with 'pixel_values'
                   - Single: [batch_size, channels, height, width]
                   - Multiple: [batch_size, num_images, channels, height, width]
        
        Returns:
            Visual features: [batch_size, hidden_size]
        """
        pixel_values = inputs['pixel_values']
        
        # Check if we have multiple images per sample
        if pixel_values.dim() == 5:
            # Multiple images: [batch_size, num_images, C, H, W]
            batch_size, num_images, C, H, W = pixel_values.shape
            
            # Flatten to process all images together
            pixel_values_flat = pixel_values.view(batch_size * num_images, C, H, W)
            
            # Encode all images
            outputs = self.encoder(pixel_values=pixel_values_flat)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [batch*num_images, hidden]
            
            # Reshape back to [batch_size, num_images, hidden]
            embeddings = embeddings.view(batch_size, num_images, -1)
            
            # Project first, then aggregate (ensures correct hidden_size for aggregator)
            projected = self.proj(embeddings.view(batch_size * num_images, -1))
            projected = projected.view(batch_size, num_images, -1)
            
            # Aggregate embeddings using the aggregator
            return self.aggregator(projected)
        else:
            # Single image: [batch_size, C, H, W]
            outputs = self.encoder(**inputs)
            last_hidden_state = outputs.last_hidden_state[:, 0, :]
            return self.proj(last_hidden_state)


class Classifier(BaseClassifier):
    def __init__(self, hidden_size: int, num_classes: int):
        super(Classifier, self).__init__()
        self.fc_in = nn.Linear(hidden_size*2, hidden_size)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size, num_classes)
    def forward(self, vis_features: Tensor, text_features: Tensor) -> Tensor:
        x = torch.cat((vis_features, text_features), dim=1)
        x = self.fc_in(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x
    

class SimpleVQA(BaseVQA):
    def __init__(self, config: SimpleVQAConfig):
        super(SimpleVQA, self).__init__()
        self.vis_encoder = VisEncoder(
            config.vis_model_name, 
            config.hidden_size,
            aggregation=config.vis_aggregation,
            aggregation_kwargs=config.vis_aggregation_kwargs
        )
        self.text_encoder = TextEncoder(
            config.text_model_name, 
            config.hidden_size,
            aggregation=config.text_aggregation,
            aggregation_kwargs=config.text_aggregation_kwargs
        )
        self.classifier = Classifier(
            hidden_size=config.hidden_size,
            num_classes=config.num_classes
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, 
                image: Tensor, 
                question_input_ids: Tensor, 
                question_attention_mask: Tensor, 
                labels: Optional[torch.Tensor] = None,
                ) -> dict:
        vis_features = self.vis_encoder({
            "pixel_values": image
        })
        
        text_features = self.text_encoder({
            "input_ids": question_input_ids,
            "attention_mask": question_attention_mask
        }) 
        logits = self.classifier(vis_features, text_features)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}