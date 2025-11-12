"""
Demonstration of Vietnamese Rule-Based VQA Augmentation

This script demonstrates how to use the rule-based augmentation system
for Vietnamese Visual Question Answering, inspired by:
"Data Augmentation for Visual Question Answering" (ACL 2017)
https://aclanthology.org/W17-3529.pdf
"""

import sys
sys.path.append('..')

from utils import VietnameseVQAAugmentation, create_augmented_dataset


def demo_basic_augmentation():
    """Demonstrate basic question augmentation."""
    print("="*70)
    print("BASIC AUGMENTATION DEMONSTRATION")
    print("="*70)
    
    augmentor = VietnameseVQAAugmentation(seed=42)
    
    # Sample Vietnamese VQA questions
    sample_questions = [
        "Cái gì trong ảnh này?",
        "Người này đang làm gì?",
        "Màu sắc của chiếc xe là gì?",
        "Có bao nhiêu người trong hình?",
        "Con vật này ở đâu?",
        "Cái này có màu gì?",
        "Người đó đang mặc áo màu gì?",
        "Thời tiết trong ảnh như thế nào?",
        "Cô gái này đang làm gì?",
        "Có phải đây là một chiếc ô tô đỏ không?",
    ]
    
    print("\nGenerating augmented questions:\n")
    
    for i, question in enumerate(sample_questions, 1):
        print(f"{i}. Original: {question}")
        
        # Generate 3 augmented versions
        augmented = augmentor.augment_question(question, num_augmentations=3)
        
        for j, aug_q in enumerate(augmented, 1):
            print(f"   Aug {j}: {aug_q}")
        print()


def demo_augmentation_strategies():
    """Demonstrate different augmentation strategies."""
    print("="*70)
    print("AUGMENTATION STRATEGIES DEMONSTRATION")
    print("="*70)
    
    augmentor = VietnameseVQAAugmentation(seed=42)
    
    test_cases = [
        {
            "question": "Cái gì trong ảnh này?",
            "strategy": "Question word replacement",
            "description": "Replaces 'gì' with 'thứ gì', 'điều gì', etc."
        },
        {
            "question": "Chiếc xe này có màu đỏ không?",
            "strategy": "Synonym replacement (color)",
            "description": "Replaces 'đỏ' with 'đỏ thẫm', 'son'"
        },
        {
            "question": "Người này đang làm gì?",
            "strategy": "Verb synonym replacement",
            "description": "Replaces 'làm' with 'thực hiện', 'tiến hành'"
        },
        {
            "question": "Con chó này ở đâu?",
            "strategy": "Demonstrative pronoun",
            "description": "Replaces 'này' with 'đây', 'nầy'"
        },
    ]
    
    print("\nTesting specific augmentation strategies:\n")
    
    for case in test_cases:
        print(f"Question: {case['question']}")
        print(f"Strategy: {case['strategy']}")
        
        augmented = augmentor.augment_question(case['question'], num_augmentations=5)
        
        print("Augmented versions:")
        for aug in augmented:
            print(f"  - {aug}")
        print()


def demo_dataset_augmentation():
    """Demonstrate augmenting a complete dataset."""
    print("="*70)
    print("DATASET AUGMENTATION DEMONSTRATION")
    print("="*70)
    
    # Simulate a small VQA dataset
    original_dataset = [
        {"question": "Cái gì trong ảnh?", "answer": "con mèo", "img_id": "000001"},
        {"question": "Người này đang làm gì?", "answer": "đọc sách", "img_id": "000002"},
        {"question": "Màu của chiếc xe là gì?", "answer": "đỏ", "img_id": "000003"},
        {"question": "Có bao nhiêu con chó?", "answer": "hai", "img_id": "000004"},
        {"question": "Thời tiết như thế nào?", "answer": "nắng", "img_id": "000005"},
    ]
    
    print(f"\nOriginal dataset size: {len(original_dataset)}")
    print("\nOriginal questions:")
    for i, item in enumerate(original_dataset, 1):
        print(f"{i}. {item['question']} → {item['answer']}")
    
    # Augment dataset with 2 variations per question
    augmented_dataset = create_augmented_dataset(
        original_dataset, 
        num_augmentations=2,
        seed=42
    )
    
    print(f"\nAugmented dataset size: {len(augmented_dataset)}")
    print(f"Growth: {len(augmented_dataset) - len(original_dataset)} new samples")
    
    print("\nAll questions (original + augmented):")
    for i, item in enumerate(augmented_dataset, 1):
        is_aug = item.get('augmented', False)
        marker = "[AUG]" if is_aug else "[ORG]"
        print(f"{i}. {marker} {item['question']} → {item['answer']}")


def demo_statistics():
    """Show statistics about augmentation potential."""
    print("="*70)
    print("AUGMENTATION STATISTICS")
    print("="*70)
    
    augmentor = VietnameseVQAAugmentation(seed=42)
    
    # Sample questions for analysis
    sample_questions = [
        "Cái gì trong ảnh này?",
        "Người này đang làm gì?",
        "Màu sắc của chiếc xe là gì?",
        "Có bao nhiêu người trong hình?",
        "Con vật này ở đâu?",
        "Cái này có màu gì?",
        "Người đó đang mặc áo màu gì?",
        "Thời tiết trong ảnh như thế nào?",
        "Cô gái này đang làm gì?",
        "Có phải đây là một chiếc ô tô đỏ không?",
    ]
    
    stats = augmentor.get_statistics(sample_questions)
    
    print("\nAugmentation Potential Analysis:")
    print(f"Total questions analyzed: {stats['total_questions']}")
    print(f"Questions with replaceable question words: {stats['questions_with_question_words']}")
    print(f"Questions with replaceable synonyms: {stats['questions_with_synonyms']}")
    print(f"Questions with demonstratives: {stats['questions_with_demonstratives']}")
    print(f"Total replaceable words found: {stats['total_replaceable_words']}")
    
    avg_replaceable = stats['total_replaceable_words'] / stats['total_questions']
    print(f"\nAverage replaceable words per question: {avg_replaceable:.2f}")
    
    # Calculate augmentation coverage
    augmentable = (stats['questions_with_question_words'] + 
                   stats['questions_with_synonyms'] + 
                   stats['questions_with_demonstratives'])
    coverage = (augmentable / (stats['total_questions'] * 3)) * 100
    print(f"Augmentation coverage: {coverage:.1f}%")


def demo_rule_categories():
    """Display all available augmentation rules."""
    print("="*70)
    print("AVAILABLE AUGMENTATION RULES")
    print("="*70)
    
    augmentor = VietnameseVQAAugmentation(seed=42)
    
    print("\n1. Question Word Variations:")
    for word, variations in sorted(augmentor.question_words.items()):
        print(f"   {word:15} → {', '.join(variations)}")
    
    print("\n2. Color Synonyms:")
    for color, synonyms in sorted(augmentor.color_synonyms.items()):
        print(f"   {color:15} → {', '.join(synonyms)}")
    
    print("\n3. Verb Synonyms:")
    for verb, synonyms in sorted(augmentor.verb_synonyms.items()):
        print(f"   {verb:15} → {', '.join(synonyms)}")
    
    print("\n4. Demonstrative Pronouns:")
    for demo, variations in sorted(augmentor.demonstratives.items()):
        print(f"   {demo:15} → {', '.join(variations)}")
    
    print("\n5. Adjective Synonyms:")
    for adj, synonyms in sorted(augmentor.adjective_synonyms.items()):
        print(f"   {adj:15} → {', '.join(synonyms)}")
    
    print("\n6. Question Starters:")
    for starter, variations in sorted(augmentor.question_starters.items()):
        print(f"   {starter:15} → {', '.join(variations)}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("VIETNAMESE VQA RULE-BASED AUGMENTATION DEMONSTRATION")
    print("Inspired by: Data Augmentation for Visual Question Answering (ACL 2017)")
    print("="*70 + "\n")
    
    # Run all demos
    demo_basic_augmentation()
    print("\n")
    
    demo_augmentation_strategies()
    print("\n")
    
    demo_dataset_augmentation()
    print("\n")
    
    demo_statistics()
    print("\n")
    
    demo_rule_categories()
    print("\n")
    
    print("="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nUsage in your code:")
    print("""
    from utils import VietnameseVQAAugmentation
    
    # Initialize augmentor
    augmentor = VietnameseVQAAugmentation(seed=42)
    
    # Augment a single question
    question = "Cái gì trong ảnh này?"
    augmented = augmentor.augment_question(question, num_augmentations=3)
    
    # Augment a dataset
    from utils import create_augmented_dataset
    augmented_dataset = create_augmented_dataset(
        original_data, 
        num_augmentations=2
    )
    """)


if __name__ == "__main__":
    main()
