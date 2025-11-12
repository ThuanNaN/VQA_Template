"""
Rule-based data augmentation for Vietnamese Visual Question Answering.

Inspired by: "Data Augmentation for Visual Question Answering" (ACL 2017)
Paper: https://aclanthology.org/W17-3529.pdf

This module implements linguistic rules for Vietnamese question paraphrasing
to augment VQA training data.
"""

import random
from typing import List, Dict, Set


class VietnameseVQAAugmentation:
    """
    Rule-based augmentation for Vietnamese VQA questions.
    
    Implements various linguistic transformations specific to Vietnamese:
    - Question word variations
    - Synonym replacement
    - Word order variations
    - Demonstrative pronoun variations
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the augmentation module.
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        
        # Question word variations in Vietnamese
        self.question_words = {
            "gì": ["cái gì", "thứ gì", "điều gì"],
            "cái gì": ["gì", "thứ gì"],
            "ai": ["người nào"],
            "đâu": ["nơi nào", "chỗ nào", "ở đâu"],
            "ở đâu": ["đâu", "nơi nào", "chỗ nào"],
            "nơi nào": ["đâu", "ở đâu", "chỗ nào"],
            "khi nào": ["lúc nào", "bao giờ"],
            "lúc nào": ["khi nào", "bao giờ"],
            "như thế nào": ["ra sao", "thế nào"],
            "bao nhiêu": ["mấy"],
            "mấy": ["bao nhiêu"],
        }
        
        # Color synonyms
        self.color_synonyms = {
            "đỏ": ["đỏ thẫm", "son"],
            "xanh": ["xanh lam", "xanh dương"],
            "trắng": ["trắng tinh", "bạch"],
            "đen": ["đen tuyền", "đen sẫm"],
            "vàng": ["vàng óng", "kim hoàng"],
            "xanh lá": ["xanh lục", "lục"],
        }
        
        # Common VQA verbs and their variations
        self.verb_synonyms = {
            "có": ["có phải", "có phải là"],
            "là": ["có phải là", "có phải"],
            "đang": ["đang làm", "đang thực hiện"],
            "làm": ["thực hiện", "tiến hành"],
            "nhìn": ["quan sát", "xem"],
            "thấy": ["nhìn thấy", "trông thấy"],
            "mặc": ["khoác", "mang"],
        }
        
        # Demonstrative pronouns
        self.demonstratives = {
            "này": ["đây", "nầy"],
            "đây": ["này", "nầy"],
            "kia": ["đó", "ấy"],
            "đó": ["kia", "ấy"],
        }
        
        # Common adjectives with synonyms
        self.adjective_synonyms = {
            "lớn": ["to", "rộng"],
            "to": ["lớn", "rộng"],
            "nhỏ": ["bé", "tí hon"],
            "bé": ["nhỏ", "tí hon"],
            "đẹp": ["xinh", "đẹp đẽ", "xinh đẹp"],
            "xấu": ["xấu xí", "không đẹp"],
            "cao": ["cao lớn"],
            "thấp": ["lùn", "thấp bé"],
        }
        
        # Question patterns - beginning of questions
        self.question_starters = {
            "có phải": ["có", "liệu có"],
            "có": ["có phải", "liệu"],
            "trong ảnh": ["trong hình", "ở trong ảnh", "ở trong hình"],
            "trong hình": ["trong ảnh", "ở trong hình", "ở trong ảnh"],
        }
        
        # All synonym dictionaries combined
        self.all_synonyms = {
            **self.color_synonyms,
            **self.verb_synonyms,
            **self.demonstratives,
            **self.adjective_synonyms,
            **self.question_starters,
        }
    
    def augment_question(self, question: str, num_augmentations: int = 1) -> List[str]:
        """
        Generate augmented versions of a Vietnamese question.
        
        Args:
            question: Original Vietnamese question
            num_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented questions (not including the original)
        """
        augmented = set()
        
        # Try different augmentation strategies
        strategies = [
            self._replace_question_words,
            self._replace_synonyms,
            self._replace_demonstratives,
            self._combine_strategies,
        ]
        
        attempts = 0
        max_attempts = num_augmentations * 10  # Prevent infinite loops
        
        while len(augmented) < num_augmentations and attempts < max_attempts:
            attempts += 1
            strategy = random.choice(strategies)
            new_question = strategy(question)
            
            # Only add if different from original and not already in set
            if new_question != question and new_question not in augmented:
                augmented.add(new_question)
        
        return list(augmented)
    
    def _replace_question_words(self, question: str) -> str:
        """Replace question words with their variations."""
        words = question.split()
        
        # Try to find and replace question words
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in self.question_words:
                alternatives = self.question_words[word_lower]
                if alternatives:
                    replacement = random.choice(alternatives)
                    # Preserve original capitalization
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    words[i] = replacement
                    break
        
        # Also try bigrams for multi-word question words
        for i in range(len(words) - 1):
            bigram = f"{words[i].lower()} {words[i+1].lower()}"
            if bigram in self.question_words:
                alternatives = self.question_words[bigram]
                if alternatives:
                    replacement = random.choice(alternatives)
                    # Preserve original capitalization
                    if words[i][0].isupper():
                        replacement = replacement.capitalize()
                    # Replace the bigram with the alternative
                    words[i] = replacement
                    words.pop(i + 1)
                    break
        
        return " ".join(words)
    
    def _replace_synonyms(self, question: str) -> str:
        """Replace words with synonyms from various categories."""
        words = question.split()
        
        # Find replaceable words
        replaceable_indices = []
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in self.all_synonyms:
                replaceable_indices.append(i)
        
        # Replace one random word if possible
        if replaceable_indices:
            idx = random.choice(replaceable_indices)
            word_lower = words[idx].lower()
            alternatives = self.all_synonyms[word_lower]
            if alternatives:
                replacement = random.choice(alternatives)
                # Preserve original capitalization
                if words[idx][0].isupper():
                    replacement = replacement.capitalize()
                words[idx] = replacement
        
        return " ".join(words)
    
    def _replace_demonstratives(self, question: str) -> str:
        """Replace demonstrative pronouns."""
        words = question.split()
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in self.demonstratives:
                alternatives = self.demonstratives[word_lower]
                if alternatives:
                    replacement = random.choice(alternatives)
                    # Preserve original capitalization
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    words[i] = replacement
                    break
        
        return " ".join(words)
    
    def _combine_strategies(self, question: str) -> str:
        """Apply multiple augmentation strategies."""
        # Apply 2-3 strategies in sequence
        num_strategies = random.randint(2, 3)
        strategies = [
            self._replace_question_words,
            self._replace_synonyms,
            self._replace_demonstratives,
        ]
        
        selected_strategies = random.sample(strategies, num_strategies)
        
        result = question
        for strategy in selected_strategies:
            result = strategy(result)
            # If no change, try another
            if result == question:
                continue
        
        return result
    
    def augment_dataset(
        self, 
        questions: List[str], 
        num_augmentations: int = 1
    ) -> Dict[int, List[str]]:
        """
        Augment a list of questions.
        
        Args:
            questions: List of original questions
            num_augmentations: Number of augmentations per question
            
        Returns:
            Dictionary mapping original question index to list of augmented questions
        """
        augmented_data = {}
        
        for idx, question in enumerate(questions):
            augmented_data[idx] = self.augment_question(question, num_augmentations)
        
        return augmented_data
    
    def get_statistics(self, questions: List[str]) -> Dict[str, int]:
        """
        Get statistics about augmentation potential.
        
        Args:
            questions: List of questions to analyze
            
        Returns:
            Dictionary with statistics about augmentable words
        """
        stats = {
            "total_questions": len(questions),
            "questions_with_question_words": 0,
            "questions_with_synonyms": 0,
            "questions_with_demonstratives": 0,
            "total_replaceable_words": 0,
        }
        
        for question in questions:
            words = question.lower().split()
            
            has_question_word = any(w in self.question_words for w in words)
            has_synonym = any(w in self.all_synonyms for w in words)
            has_demonstrative = any(w in self.demonstratives for w in words)
            
            if has_question_word:
                stats["questions_with_question_words"] += 1
            if has_synonym:
                stats["questions_with_synonyms"] += 1
            if has_demonstrative:
                stats["questions_with_demonstratives"] += 1
            
            replaceable = sum(1 for w in words if w in self.all_synonyms or w in self.question_words)
            stats["total_replaceable_words"] += replaceable
        
        return stats


def create_augmented_dataset(
    original_data: List[Dict[str, str]], 
    num_augmentations: int = 1,
    seed: int = 42
) -> List[Dict[str, str]]:
    """
    Create an augmented dataset from original VQA data.
    
    Args:
        original_data: List of dictionaries with 'question', 'answer', 'img_id' keys
        num_augmentations: Number of augmented questions per original
        seed: Random seed
        
    Returns:
        Extended dataset with original + augmented questions
    """
    augmentor = VietnameseVQAAugmentation(seed=seed)
    augmented_dataset = list(original_data)  # Keep originals
    
    for item in original_data:
        question = item['question']
        augmented_questions = augmentor.augment_question(question, num_augmentations)
        
        # Add augmented versions with same answer and image
        for aug_question in augmented_questions:
            augmented_item = {
                'question': aug_question,
                'answer': item['answer'],
                'img_id': item['img_id'],
                'augmented': True,  # Mark as augmented
            }
            augmented_dataset.append(augmented_item)
    
    return augmented_dataset
