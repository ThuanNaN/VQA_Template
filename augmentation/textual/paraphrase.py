"""
Paraphrase-based text augmentation for VQA.

This augmentation generates multiple paraphrases of the input question
and returns them as a list for ensemble encoding.
"""

from typing import List, Optional
from ..base import BaseTextAugmentation


class ParaphraseTextAugmentation(BaseTextAugmentation):
    """
    Generate multiple paraphrases of input text for ensemble encoding.
    
    This augmentation can work in two modes:
    1. Online: Generate paraphrases on-the-fly using LLM (slower, more diverse)
    2. Offline: Load pre-generated paraphrases from cache (faster, recommended)
    
    Returns a list of paraphrased texts that will be encoded and aggregated.
    """
    
    def __init__(self, 
                 difficulty: float = 0.5,
                 num_paraphrases: int = 3,
                 include_original: bool = True,
                 llm_client=None,
                 paraphrase_cache: Optional[dict] = None):
        """
        Initialize paraphrase augmentation.
        
        Args:
            difficulty: Not used for paraphrasing, kept for consistency
            num_paraphrases: Number of paraphrases to generate (excluding original)
            include_original: Whether to include original text in output
            llm_client: Optional LLM client for online paraphrase generation
            paraphrase_cache: Optional dict mapping original text to paraphrases
        """
        super().__init__(difficulty)
        self.num_paraphrases = num_paraphrases
        self.include_original = include_original
        self.llm_client = llm_client
        self.paraphrase_cache = paraphrase_cache or {}
    
    def _configure_parameters(self):
        """Configure parameters based on difficulty (not used for paraphrasing)."""
        pass
    
    def augment(self, text: str) -> List[str]:
        """
        Generate paraphrases of input text.
        
        Args:
            text: Original question text
            
        Returns:
            List of paraphrased texts (including original if specified)
        """
        paraphrases = []
        
        # Add original if specified
        if self.include_original:
            paraphrases.append(text)
        
        # Try to get from cache first
        if text in self.paraphrase_cache:
            cached = self.paraphrase_cache[text]
            paraphrases.extend(cached[:self.num_paraphrases])
        
        # If not enough paraphrases, generate online (if LLM available)
        elif self.llm_client is not None:
            for _ in range(self.num_paraphrases):
                try:
                    prompt = f"Paraphrase the following Vietnamese question while keeping the same meaning:\n{text}\n\nParaphrase:"
                    paraphrase = self.llm_client.generate(prompt).strip()
                    paraphrases.append(paraphrase)
                except Exception as e:
                    # Fallback to original if generation fails
                    print(f"Warning: Paraphrase generation failed: {e}")
                    paraphrases.append(text)
        else:
            # No cache and no LLM, just return original multiple times
            paraphrases.extend([text] * self.num_paraphrases)
        
        return paraphrases


class SimpleParaphraseAugmentation(BaseTextAugmentation):
    """
    Simple rule-based paraphrase augmentation for Vietnamese.
    
    Uses simple transformations to create paraphrases:
    - Synonym replacement
    - Word reordering (for certain patterns)
    - Adding/removing politeness markers
    
    Returns a list of paraphrased texts.
    """
    
    def __init__(self, difficulty: float = 0.5, num_paraphrases: int = 2):
        """
        Initialize simple paraphrase augmentation.
        
        Args:
            difficulty: Controls variation strength (0.0 to 1.0)
            num_paraphrases: Number of paraphrases to generate
        """
        super().__init__(difficulty)
        self.num_paraphrases = num_paraphrases
        self._configure_parameters()
    
    def _configure_parameters(self):
        """Configure synonym replacement rate based on difficulty."""
        self.replacement_rate = 0.1 + (self.difficulty * 0.4)  # 0.1 to 0.5
    
    def augment(self, text: str) -> List[str]:
        """
        Generate simple paraphrases of input text.
        
        Args:
            text: Original question text
            
        Returns:
            List of paraphrased texts (including original)
        """
        import random
        
        paraphrases = [text]  # Always include original
        
        # Simple Vietnamese question word replacements
        replacements = {
            'cái gì': ['gì', 'những gì'],
            'ở đâu': ['đâu', 'nơi nào'],
            'bao nhiêu': ['mấy'],
            'như thế nào': ['ra sao', 'thế nào'],
            'tại sao': ['vì sao', 'sao'],
            'có phải': ['có'],
            'màu gì': ['màu sắc nào', 'màu sắc gì'],
        }
        
        for _ in range(self.num_paraphrases):
            paraphrase = text
            
            # Try replacements
            for original, alternatives in replacements.items():
                if original in paraphrase and random.random() < self.replacement_rate:
                    alt = random.choice(alternatives)
                    paraphrase = paraphrase.replace(original, alt, 1)
            
            paraphrases.append(paraphrase)
        
        return paraphrases
