from typing import Optional
import torch
from torch import nn, Tensor
from .base import (
    BaseTextEncoder,
    BaseVisEncoder,
    BaseClassifier,
    BaseVQA
)
from transformers import AutoModel
from dataclasses import dataclass

@dataclass
class SimpleVQAConfig:
    vis_model_name: str
    text_model_name: str
    hidden_size: int
    num_classes: int

class TextEncoder(BaseTextEncoder):
    def __init__(self, model_name: str):
        super(TextEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, inputs: str) -> Tensor:
        outputs = self.encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :] # [batch_size, hidden_size]


class VisEncoder(BaseVisEncoder):
    def __init__(self, model_name: str):
        super(VisEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :] # [batch_size, hidden_size]


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
        self.vis_encoder = VisEncoder(config.vis_model_name)
        self.text_encoder = TextEncoder(config.text_model_name)
        assert self.vis_encoder.encoder.config.hidden_size == \
            self.text_encoder.encoder.config.hidden_size, \
            "Hidden size of text and visual encoder should be same"
        
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
                ) -> Tensor:
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