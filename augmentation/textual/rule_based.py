"""
POS Tagging-based Rule-Based Paraphrase Augmentation for Vietnamese.

This module implements text augmentation using POS (Part-of-Speech) tagging
to intelligently paraphrase Vietnamese questions using multiple rules:
1. Synonym replacement by POS tags
2. ADV (adverb) movement to front
3. Active to Passive voice transformation
"""

import random
from typing import Optional, Dict, Any, Union, List
from ..base import BaseTextAugmentation

try:
    from underthesea import pos_tag, word_tokenize
except ImportError:
    pos_tag = None
    word_tokenize = None
    print("Warning: underthesea not installed. Install with: pip install underthesea")


class RuleBasedTextAugmentation(BaseTextAugmentation):
    """
    POS tagging-based paraphrase augmentation for Vietnamese.
    
    This augmentation uses POS (Part-of-Speech) tagging from underthesea library
    to intelligently paraphrase Vietnamese questions using three main rules:
    
    Rule 1: Synonym replacement by POS tags
    - Replaces words with synonyms based on their POS category (V, N, A, etc.)
    
    Rule 2: ADV (Adverb) movement
    - Moves adverbs to the front of the sentence for stylistic variation
    
    Rule 3: Active to Passive transformation
    - Converts active voice to passive voice: Subject + V + Object → Object + được + Subject + V
    
    Three-phase curriculum learning strategy (matching visual augmentation):
    - Easy (0.0 - 0.33): Generate 2 paraphrases (total 3 with original)
    - Medium (0.33 - 0.66): Generate 1 paraphrase (total 2 with original)
    - Hard (0.66 - 1.0): No augmentation (only original)
    
    POS Tags used (from underthesea):
    - V: Verb
    - N: Noun
    - A: Adjective
    - P: Pronoun
    - R: Adverb
    - L: Determiner
    - M: Numeral
    - E: Preposition
    - C: Conjunction
    """
    
    def __init__(self, difficulty: Union[int, float] = 0.0, seed: Optional[int] = None):
        super().__init__(difficulty, seed)
        self.rng = random.Random(seed)
        
        if pos_tag is None or word_tokenize is None:
            raise ImportError(
                "underthesea library is required for POS tagging. "
                "Install with: pip install underthesea"
            )
        
        self._init_synonym_dict()
        self._configure_parameters()
    
    def _init_synonym_dict(self) -> None:
        """Initialize synonym dictionaries organized by POS tags."""
        
        # Synonym dictionary by POS tags
        self.SYNONYM_DICT = {
            "V": {  # Verbs
                "giải thích": ["trình bày", "diễn giải"],
                "hoàn thành": ["kết thúc", "làm xong"],
                "xem": ["nhìn", "quan sát"],
                "làm": ["thực hiện", "tiến hành"],
                "thực hiện": ["làm", "tiến hành"],
                "tiến hành": ["làm", "thực hiện"],
                "đứng": ["dựng", "đứng lên"],
                "ngồi": ["ngồi xuống"],
                "nhìn": ["nhìn thấy", "trông thấy", "quan sát"],
                "nhìn thấy": ["nhìn", "trông thấy"],
                "trông thấy": ["nhìn", "nhìn thấy"],
                "quan sát": ["nhìn", "xem"],
                "mặc": ["mang", "đeo"],
                "mang": ["mặc", "đeo"],
                "đeo": ["mang", "mặc"],
            },
            "A": {  # Adjectives
                "đẹp": ["xinh", "xinh đẹp"],
                "rõ ràng": ["minh bạch", "dễ hiểu"],
                "lớn": ["to", "rộng"],
                "to": ["lớn", "rộng"],
                "rộng": ["lớn", "to"],
                "nhỏ": ["bé", "tí"],
                "bé": ["nhỏ", "tí"],
                "tí": ["nhỏ", "bé"],
                "cao": ["cao lớn"],
                "thấp": ["lùn"],
                "xinh": ["đẹp", "đẹp đẽ"],
                "xấu": ["xấu xí"],
                "đỏ": ["đỏ thẫm"],
                "xanh": ["xanh lam", "xanh dương"],
                "vàng": ["vàng óng"],
                "trắng": ["trắng tinh"],
                "đen": ["đen tuyền"],
            },
            "N": {  # Nouns
                "ngôi nhà": ["căn nhà", "ngôi biệt thự"],
                "phim": ["bộ phim", "tác phẩm điện ảnh"],
                "màu": ["màu sắc", "sắc"],
                "màu sắc": ["màu", "sắc"],
                "người": ["con người", "cá nhân"],
                "vật": ["đồ vật", "vật thể"],
                "nơi": ["chỗ", "địa điểm"],
                "chỗ": ["nơi", "địa điểm"],
            },
            "P": {  # Pronouns
                "gì": ["cái gì", "thứ gì"],
                "cái gì": ["gì", "thứ gì"],
                "thứ gì": ["gì", "cái gì"],
                "đâu": ["nơi nào", "chỗ nào"],
                "nơi nào": ["đâu", "chỗ nào"],
                "chỗ nào": ["đâu", "nơi nào"],
                "ai": ["người nào"],
                "người nào": ["ai"],
                "bao nhiêu": ["mấy"],
                "mấy": ["bao nhiêu"],
            },
            "R": {  # Adverbs
                "rất": ["cực kỳ", "vô cùng"],
                "cực kỳ": ["rất", "vô cùng"],
                "vô cùng": ["rất", "cực kỳ"],
            },
            "L": {  # Determiners
                "này": ["đây", "nầy"],
                "đây": ["này", "nầy"],
                "kia": ["đó", "ấy"],
                "đó": ["kia", "ấy"],
            }
        }
    
    def _configure_parameters(self) -> None:
        """
        Configure augmentation parameters based on difficulty (0.0-1.0).
        
        Three-phase strategy (matching visual augmentation):
        - Easy (0.0 - 0.33): Generate 2 paraphrases (total 3 with original)
        - Medium (0.33 - 0.66): Generate 1 paraphrase (total 2 with original)
        - Hard (0.66 - 1.0): No augmentation (only original)
        """
        # Determine augmentation phase and number of paraphrases
        if self.difficulty < 0.33:
            # Easy: Generate 2 paraphrases
            self.augmentation_phase = 'easy'
            self.num_paraphrases = 2
        elif self.difficulty < 0.66:
            # Medium: Generate 1 paraphrase
            self.augmentation_phase = 'medium'
            self.num_paraphrases = 1
        else:
            # Hard: No augmentation
            self.augmentation_phase = 'hard'
            self.num_paraphrases = 0
    
    def augment(self, text: str, **kwargs) -> List[str]:
        """
        Apply POS-based paraphrase augmentation using multiple rules.
        
        Args:
            text: Input text (question) to augment
            **kwargs: Additional options
            
        Returns:
            List of text strings including original and paraphrases.
            Always returns consistent length:
            - Easy: [original, para1, para2] (3 total)
            - Medium: [original, para1] (2 total)
            - Hard: [original] (1 total)
            
            If not enough paraphrases can be generated, duplicates the original.
        """
        # Hard phase: no augmentation, return only original
        if self.num_paraphrases == 0:
            return [text]
        
        # Generate all possible paraphrases
        all_paraphrases = self._paraphrase(text)
        
        # Select paraphrases based on difficulty
        if not all_paraphrases:
            # No paraphrases generated, pad with duplicates of original
            # This ensures consistent batch sizes
            return [text] * (self.num_paraphrases + 1)
        
        # Select num_paraphrases from available paraphrases
        if len(all_paraphrases) >= self.num_paraphrases:
            selected_paraphrases = self.rng.sample(all_paraphrases, self.num_paraphrases)
        else:
            # Not enough unique paraphrases, use what we have and pad with original
            selected_paraphrases = all_paraphrases
            # Pad with original text to reach desired count
            while len(selected_paraphrases) < self.num_paraphrases:
                selected_paraphrases.append(text)
        
        # Return original + paraphrases (always num_paraphrases + 1 items)
        return [text] + selected_paraphrases
    
    def _paraphrase(self, sentence: str) -> List[str]:
        """
        Generate paraphrases using all rules.
        
        Args:
            sentence: Input sentence
            
        Returns:
            List of unique paraphrased sentences
        """
        candidates = []
        
        # Apply all paraphrase rules
        candidates += self._rule_synonym(sentence)
        candidates += self._rule_move_adv(sentence)
        candidates += self._rule_active_to_passive(sentence)
        
        # Remove duplicates and original sentence
        unique_candidates = list(set([c for c in candidates if c != sentence]))
        
        return unique_candidates
    
    def _rule_synonym(self, sentence: str) -> List[str]:
        """
        Rule 1: Synonym replacement by POS tags.
        
        Replaces words with their synonyms based on POS category.
        
        Args:
            sentence: Input sentence
            
        Returns:
            List of paraphrased sentences with synonym replacements
        """
        try:
            tokens = word_tokenize(sentence, format="text")
            tagged = pos_tag(tokens)
        except Exception as e:
            print(f"Warning: POS tagging failed: {e}")
            return []
        
        new_sentences = []
        
        for i, (word, pos) in enumerate(tagged):
            # Get main POS category (e.g., V, N, A)
            pos_main = pos.split("_")[0] if "_" in pos else pos
            
            # Check if word has synonyms in this POS category
            if pos_main in self.SYNONYM_DICT and word in self.SYNONYM_DICT[pos_main]:
                for syn in self.SYNONYM_DICT[pos_main][word]:
                    new_tokens = tokens.split()
                    if i < len(new_tokens):
                        new_tokens[i] = syn
                        new_sentences.append(" ".join(new_tokens))
        
        return new_sentences
    
    def _rule_move_adv(self, sentence: str) -> List[str]:
        """
        Rule 2: Move ADV (adverb) to front.
        
        Moves adverbs to the beginning of the sentence for stylistic variation.
        Pattern: ... ADV ... → ADV, ...
        
        Args:
            sentence: Input sentence
            
        Returns:
            List with sentence having adverb moved to front (or empty list)
        """
        try:
            tokens = word_tokenize(sentence, format="text").split()
            tagged = pos_tag(" ".join(tokens))
        except Exception as e:
            print(f"Warning: POS tagging failed: {e}")
            return []
        
        for i, (word, pos) in enumerate(tagged):
            # R = Adverb in underthesea
            if pos.startswith("R") and i > 0:  # Don't move if already at front
                adv = word
                remaining = tokens[:i] + tokens[i+1:]
                # Capitalize adverb and add comma
                paraphrased = adv.capitalize() + ", " + " ".join(remaining)
                # Lowercase the first letter of remaining if it was lowercase
                if remaining and remaining[0][0].islower():
                    paraphrased = adv.capitalize() + ", " + remaining[0].lower() + " " + " ".join(remaining[1:])
                return [paraphrased]
        
        return []
    
    def _rule_active_to_passive(self, sentence: str) -> List[str]:
        """
        Rule 3: Active to Passive transformation (basic).
        
        Converts active voice to passive voice.
        Pattern: Subject + V + Object → Object + được + Subject + V
        
        Args:
            sentence: Input sentence
            
        Returns:
            List with passive voice transformation (or empty list)
        """
        try:
            tokens = word_tokenize(sentence, format="text").split()
            tagged = pos_tag(" ".join(tokens))
        except Exception as e:
            print(f"Warning: POS tagging failed: {e}")
            return []
        
        # Find first verb
        verb_idx = None
        for i, (w, p) in enumerate(tagged):
            if p.startswith("V"):
                verb_idx = i
                break
        
        # Need verb in middle of sentence with subject and object
        if verb_idx is None or verb_idx == 0 or verb_idx == len(tokens) - 1:
            return []
        
        subject = tokens[:verb_idx]
        verb = tokens[verb_idx]
        obj = tokens[verb_idx + 1:]
        
        # Build passive: Object + được + Subject + Verb
        passive = obj + ["được"] + subject + [verb]
        
        # Capitalize first word
        if passive:
            passive[0] = passive[0][0].upper() + passive[0][1:] if len(passive[0]) > 0 else passive[0]
        
        return [" ".join(passive)]
    
    def get_augmentation_info(self) -> Dict[str, Any]:
        """Get augmentation configuration info."""
        total_synonyms = sum(len(words) for pos_dict in self.SYNONYM_DICT.values() for words in pos_dict.values())
        return {
            'type': 'POSBasedParaphraseAugmentation',
            'method': 'rule_based_pos_tagging',
            'rules': ['synonym_replacement', 'adv_movement', 'active_to_passive'],
            'difficulty': self.difficulty,
            'augmentation_phase': self.augmentation_phase,
            'num_paraphrases': self.num_paraphrases,
            'total_outputs': self.num_paraphrases + 1,  # original + paraphrases
            'total_synonyms': total_synonyms,
            'pos_categories': list(self.SYNONYM_DICT.keys()),
            'seed': self.seed
        }
