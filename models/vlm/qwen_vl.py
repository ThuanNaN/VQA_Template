"""
Qwen2-VL model wrapper for VQA task.
Supports loading with LoRA/QLoRA configuration.
"""

import torch
from typing import Optional, Dict, Any, Tuple

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel


class Qwen2VLModel:
    """
    Wrapper class for Qwen2-VL model with LoRA support.
    
    Example:
        >>> qwen_model = Qwen2VLModel(
        ...     model_name="Qwen/Qwen2-VL-2B-Instruct",
        ...     use_4bit=True,
        ...     lora_r=16
        ... )
        >>> qwen_model.load_model(for_training=True)
        >>> model, processor = qwen_model.get_model_and_processor()
    """
    
    SUPPORTED_MODELS = [
        "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
    ]
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        use_4bit: bool = False,
        use_8bit: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj"),
    ):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        
        self.model = None
        self.processor = None
        self.peft_config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization config for 4-bit or 8-bit loading."""
        if self.use_4bit and self.device == "cuda":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif self.use_8bit and self.device == "cuda":
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        return None
    
    def _get_lora_config(self) -> LoraConfig:
        """Get LoRA configuration."""
        return LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            target_modules=list(self.lora_target_modules),
            task_type="CAUSAL_LM",
        )
    
    def load_model(self, for_training: bool = True) -> "Qwen2VLModel":
        """
        Load the model and processor.
        
        Args:
            for_training: If True, disable cache for training.
        
        Returns:
            Self for chaining.
        """
        quant_config = self._get_quantization_config()
        
        # Model kwargs
        model_kwargs = {
            "device_map": self.device_map if self.device == "cuda" else None,
            "use_cache": not for_training,
        }
        
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
            print(f"✅ Loading with {'4-bit' if self.use_4bit else '8-bit'} quantization")
        else:
            model_kwargs["torch_dtype"] = self.torch_dtype
            print(f"✅ Loading with dtype: {self.torch_dtype}")
        
        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Load processor
        self.processor = Qwen2VLProcessor.from_pretrained(self.model_name)
        self.processor.tokenizer.padding_side = "right"
        
        print(f"✅ Model loaded: {self.model_name}")
        print(f"   Parameters: {self.model.num_parameters():,}")
        
        return self
    
    def apply_lora(self) -> "Qwen2VLModel":
        """Apply LoRA to the model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.peft_config = self._get_lora_config()
        
        print(f"✅ Applying LoRA:")
        print(f"   r={self.lora_r}, alpha={self.lora_alpha}")
        print(f"   target_modules={self.lora_target_modules}")
        
        trainable_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.model = get_peft_model(self.model, self.peft_config)
        trainable_after = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   Trainable params: {trainable_before:,} → {trainable_after:,}")
        
        return self
    
    def load_adapter(self, adapter_path: str) -> "Qwen2VLModel":
        """Load a trained LoRA adapter."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model.load_adapter(adapter_path)
        print(f"✅ Adapter loaded from: {adapter_path}")
        
        return self
    
    def get_model_and_processor(self) -> Tuple[Any, Any]:
        """Get the model and processor."""
        return self.model, self.processor
    
    def get_peft_config(self) -> Optional[LoraConfig]:
        """Get the PEFT config for trainer."""
        return self.peft_config or self._get_lora_config()
    
    def save(self, save_path: str):
        """Save model and processor."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        print(f"✅ Model saved to: {save_path}")


def load_qwen2vl_model(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    use_4bit: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    for_training: bool = True,
    apply_lora: bool = True,
) -> Tuple[Any, Any, Optional[LoraConfig]]:
    """
    Convenience function to load Qwen2-VL model.
    
    Args:
        model_name: Model name or path
        use_4bit: Use 4-bit quantization
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        for_training: Load for training (disable cache)
        apply_lora: Apply LoRA adapter
    
    Returns:
        Tuple of (model, processor, peft_config)
    """
    qwen = Qwen2VLModel(
        model_name=model_name,
        use_4bit=use_4bit,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    qwen.load_model(for_training=for_training)
    
    peft_config = None
    if apply_lora:
        qwen.apply_lora()
        peft_config = qwen.get_peft_config()
    
    return qwen.model, qwen.processor, peft_config
