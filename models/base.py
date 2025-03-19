from torch import nn, Tensor

class BaseTextEncoder(nn.Module):
    def __init__(self):
        super(BaseTextEncoder, self).__init__()
    def forward(self, inputs: str) -> Tensor:
        raise NotImplementedError

    
class BaseVisEncoder(nn.Module):
    def __init__(self):
        super(BaseVisEncoder, self).__init__()
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError


class BaseClassifier(nn.Module):
    def __init__(self):
        super(BaseClassifier, self).__init__()
    def forward(self, vis_features: Tensor, text_features: Tensor) -> Tensor:
        raise NotImplementedError


class BaseVQA(nn.Module):
    def __init__(self):
        super(BaseVQA, self).__init__()
    def forward(self,
                image: Tensor, 
                question_input_ids: Tensor, 
                question_attention_mask: Tensor, 
                labels: Tensor | None = None
                ) -> dict:
        raise NotImplementedError

