"""
Text augmentation for VQA with Curriculum Learning support.

This module implements text augmentation strategies that can be used
with curriculum learning to progressively increase difficulty.
"""

import random
from typing import Optional, Dict, Any, Union, List
from ..base import BaseTextAugmentation

class RuleBasedTextAugmentation(BaseTextAugmentation):
    """
    Rule-based text augmentation using rich linguistic rules for Vietnamese.
    
    This augmentation strategy uses sophisticated Vietnamese linguistic rules including:
    - Question word variations (gì, cái gì, thứ gì)
    - Color synonyms (đỏ, đỏ thẫm, son)
    - Verb synonyms (có, có phải, có phải là)
    - Demonstratives (này, đây, nầy)
    - Adjective synonyms (lớn, to, rộng)
    - Vietnamese-specific paraphrasing
    
    Curriculum learning controls the FREQUENCY of augmentation using smooth difficulty (0.0-1.0):
    - 0.0 (easy): 5% of texts are augmented
    - 0.5 (medium): ~27% of texts are augmented
    - 1.0 (hard): 50% of texts are augmented
    
    Based on "Data Augmentation for Visual Question Answering" (ACL 2017)
    Paper: https://aclanthology.org/W17-3529.pdf
    """
    
    def __init__(self, difficulty: Union[int, float] = 0.0, seed: Optional[int] = None):
        super().__init__(difficulty, seed)
        self.rng = random.Random(seed)
        self._init_vietnamese_rules()
        self._configure_parameters()
    
    def _init_vietnamese_rules(self) -> None:
        """Initialize Vietnamese linguistic rules for augmentation."""
        # Question word variations
        self.question_word_variations = {
            "gì": ["cái gì", "thứ gì", "điều gì"],
            "cái gì": ["gì", "thứ gì", "điều gì"],
            "thứ gì": ["gì", "cái gì", "điều gì"],
            "điều gì": ["gì", "cái gì", "thứ gì"],
            "đâu": ["nơi nào", "chỗ nào", "ở đâu"],
            "ở đâu": ["đâu", "nơi nào", "chỗ nào"],
            "nơi nào": ["đâu", "ở đâu", "chỗ nào"],
            "chỗ nào": ["đâu", "ở đâu", "nơi nào"],
            "khi nào": ["lúc nào", "bao giờ"],
            "lúc nào": ["khi nào", "bao giờ"],
            "bao giờ": ["khi nào", "lúc nào"],
            "ai": ["người nào"],
            "người nào": ["ai"],
            "bao nhiêu": ["mấy"],
            "mấy": ["bao nhiêu"],
            "như thế nào": ["ra sao", "thế nào"],
            "thế nào": ["như thế nào", "ra sao"],
            "ra sao": ["như thế nào", "thế nào"],
        }
        
        # Color synonyms
        self.color_synonyms = {
            "đỏ": ["đỏ thẫm", "son"],
            "đỏ thẫm": ["đỏ", "son"],
            "son": ["đỏ", "đỏ thẫm"],
            "xanh": ["xanh lam", "xanh dương"],
            "xanh lam": ["xanh", "xanh dương"],
            "xanh dương": ["xanh", "xanh lam"],
            "vàng": ["vàng óng", "vàng tươi"],
            "vàng óng": ["vàng", "vàng tươi"],
            "vàng tươi": ["vàng", "vàng óng"],
            "trắng": ["trắng tinh", "trắng bóc"],
            "trắng tinh": ["trắng", "trắng bóc"],
            "trắng bóc": ["trắng", "trắng tinh"],
            "đen": ["đen tuyền", "đen thui"],
            "đen tuyền": ["đen", "đen thui"],
            "đen thui": ["đen", "đen tuyền"],
        }
        
        # Verb synonyms
        self.verb_synonyms = {
            "làm": ["thực hiện", "tiến hành"],
            "thực hiện": ["làm", "tiến hành"],
            "tiến hành": ["làm", "thực hiện"],
            "có": ["có phải", "có phải là"],
            "có phải": ["có", "có phải là"],
            "có phải là": ["có", "có phải"],
            "đứng": ["đứng lên", "dựng"],
            "đứng lên": ["đứng", "dựng"],
            "dựng": ["đứng", "đứng lên"],
            "ngồi": ["ngồi xuống"],
            "ngồi xuống": ["ngồi"],
            "nhìn": ["nhìn thấy", "trông thấy", "xem"],
            "nhìn thấy": ["nhìn", "trông thấy", "xem"],
            "trông thấy": ["nhìn", "nhìn thấy", "xem"],
            "xem": ["nhìn", "nhìn thấy", "trông thấy"],
        }
        
        # Demonstrative pronoun variations
        self.demonstrative_variations = {
            "này": ["đây", "nầy"],
            "đây": ["này", "nầy"],
            "nầy": ["này", "đây"],
            "kia": ["đó", "ấy"],
            "đó": ["kia", "ấy"],
            "ấy": ["kia", "đó"],
        }
        
        # Adjective synonyms
        self.adjective_synonyms = {
            "lớn": ["to", "rộng"],
            "to": ["lớn", "rộng"],
            "rộng": ["lớn", "to"],
            "nhỏ": ["bé", "tí"],
            "bé": ["nhỏ", "tí"],
            "tí": ["nhỏ", "bé"],
            "cao": ["cao lớn"],
            "cao lớn": ["cao"],
            "thấp": ["lùn"],
            "lùn": ["thấp"],
            "đẹp": ["xinh", "đẹp đẽ"],
            "xinh": ["đẹp", "đẹp đẽ"],
            "đẹp đẽ": ["đẹp", "xinh"],
            "xấu": ["xấu xí"],
            "xấu xí": ["xấu"],
        }
        
        # Question starters
        self.question_starters = {
            "có phải": ["có phải là", "phải chăng"],
            "có phải là": ["có phải", "phải chăng"],
            "phải chăng": ["có phải", "có phải là"],
        }
        
        # Combine all synonym dictionaries
        self.all_synonyms = {
            **self.question_word_variations,
            **self.color_synonyms,
            **self.verb_synonyms,
            **self.demonstrative_variations,
            **self.adjective_synonyms,
            **self.question_starters,
        }
    
    def _configure_parameters(self) -> None:
        """Configure augmentation frequency based on difficulty (0.0-1.0)."""
        # Smooth interpolation for apply probability: 5% to 50%
        min_prob = 0.05
        max_prob = 0.50
        self.apply_prob = min_prob + self.difficulty * (max_prob - min_prob)
        
        # Max replacements: 1 to 3 (rounded)
        min_replacements = 1
        max_replacements = 3
        self.max_replacements = int(min_replacements + self.difficulty * (max_replacements - min_replacements))
        self.max_replacements = max(1, self.max_replacements)  # At least 1
    
    def augment(self, text: str, **kwargs) -> List[str]:
        """
        Apply rule-based text augmentation with curriculum learning.
        
        Args:
            text: Input text (question) to augment
            **kwargs: Additional options (num_replacements, etc.)
            
        Returns:
            List containing single augmented text string for multi-view consistency.
            Returns [original_text] if no augmentation is applied (based on apply_prob).
        """
        # Apply augmentation based on probability (curriculum learning)
        if self.rng.random() >= self.apply_prob:
            return [text]
        
        # Perform synonym replacement
        num_replacements = kwargs.get('num_replacements', self.max_replacements)
        augmented = self._replace_synonyms(text, max_replacements=num_replacements)
        
        return [augmented]
    
    def _replace_synonyms(self, text: str, max_replacements: int = 2) -> str:
        """
        Replace words with their Vietnamese synonyms.
        
        Args:
            text: Input text
            max_replacements: Maximum number of words to replace
            
        Returns:
            Text with synonyms replaced
        """
        words = text.split()
        replacements_made = 0
        
        # Find all possible replacement positions
        replacement_candidates = []
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Check for multi-word phrases first (up to 3 words)
            for phrase_len in [3, 2, 1]:
                if i + phrase_len <= len(words):
                    phrase = ' '.join([words[j].lower() for j in range(i, i + phrase_len)])
                    if phrase in self.all_synonyms:
                        replacement_candidates.append((i, phrase_len, phrase))
                        break
        
        # Shuffle candidates for randomness
        self.rng.shuffle(replacement_candidates)
        
        # Track which positions have been modified
        modified_positions = set()
        
        # Apply replacements
        for start_idx, phrase_len, phrase in replacement_candidates:
            if replacements_made >= max_replacements:
                break
            
            # Check if any position in this phrase has been modified
            if any(pos in modified_positions for pos in range(start_idx, start_idx + phrase_len)):
                continue
            
            # Get synonym
            synonyms = self.all_synonyms[phrase]
            if synonyms:
                replacement = self.rng.choice(synonyms)
                
                # Replace the phrase
                replacement_words = replacement.split()
                for j, rep_word in enumerate(replacement_words):
                    if start_idx + j < len(words):
                        # Preserve capitalization of first word
                        if j == 0 and words[start_idx].istitle():
                            words[start_idx + j] = rep_word.capitalize()
                        else:
                            words[start_idx + j] = rep_word
                
                # If replacement is shorter, remove extra words
                if len(replacement_words) < phrase_len:
                    for _ in range(phrase_len - len(replacement_words)):
                        words.pop(start_idx + len(replacement_words))
                
                # Mark positions as modified
                for pos in range(start_idx, start_idx + max(phrase_len, len(replacement_words))):
                    modified_positions.add(pos)
                
                replacements_made += 1
        
        return ' '.join(words)
    
    def get_augmentation_info(self) -> Dict[str, Any]:
        """Get augmentation configuration info."""
        return {
            'type': 'RuleBasedTextAugmentation',
            'difficulty': self.difficulty,
            'apply_prob': self.apply_prob,
            'max_replacements': self.max_replacements,
            'num_rules': len(self.all_synonyms),
            'seed': self.seed
        }
