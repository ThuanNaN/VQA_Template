"""
Example usage of Text Augmentation with Curriculum Learning for VQA.

This script demonstrates how to use text augmentation classes
with curriculum learning to augment questions for VQA training.
"""

import sys
sys.path.append('..')

import os
import pandas as pd
from PIL import Image
import json
from pathlib import Path

from augmentation import (
    RuleBasedTextAugmentation,
    CurriculumScheduler,
    AugmentationFactory
)


def demonstrate_vivqa_augmentation_with_images():
    """
    Demonstrate text augmentation on real ViVQA dataset samples.
    Load real images and questions, apply augmentation, and save results.
    """
    print("="*60)
    print("Demonstrating Text Augmentation on Real ViVQA Dataset")
    print("="*60)
    
    # Setup paths
    vivqa_train_path = "data/vivqa/train.csv"
    vivqa_images_dir = "data/vivqa/images"
    output_dir = "runs/text_aug_demo"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if data exists
    if not os.path.exists(vivqa_train_path):
        print(f"\n⚠️  ViVQA training data not found at: {vivqa_train_path}")
        print("Please ensure the ViVQA dataset is downloaded.")
        return
    
    if not os.path.exists(vivqa_images_dir):
        print(f"\n⚠️  ViVQA images directory not found at: {vivqa_images_dir}")
        print("Please ensure the ViVQA images are downloaded.")
        return
    
    # Load ViVQA data
    print(f"\nLoading ViVQA data from: {vivqa_train_path}")
    vivqa_data = pd.read_csv(vivqa_train_path)
    print(f"Total samples in dataset: {len(vivqa_data)}")
    
    # Select a diverse sample of questions
    num_samples = 10
    sample_data = vivqa_data.head(num_samples)
    
    print(f"\nSelected {num_samples} samples for augmentation demonstration")
    
    # Demonstrate augmentation at different difficulty levels
    results = []
    
    for difficulty, name in [(0.2, "EASY"), (0.5, "MEDIUM"), (0.8, "HARD")]:
        print(f"\n{'-'*60}")
        print(f"Processing with {name} difficulty (difficulty={difficulty})")
        print(f"{'-'*60}")
        
        augmentor = RuleBasedTextAugmentation(difficulty=difficulty, seed=42)
        info = augmentor.get_augmentation_info()
        
        print(f"Configuration:")
        print(f"  - Augmentation probability: {info['apply_prob']:.0%}")
        print(f"  - Max replacements per question: {info['max_replacements']}")
        print(f"  - Number of linguistic rules: {info['num_rules']}")
        
        difficulty_results = []
        
        for idx, row in sample_data.iterrows():
            img_id = str(row['img_id']).zfill(12)
            img_path = os.path.join(vivqa_images_dir, f"{img_id}.jpg")
            
            # Check if image exists
            if not os.path.exists(img_path):
                print(f"⚠️  Image not found: {img_path}")
                continue
            
            # Load image
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"⚠️  Error loading image {img_path}: {e}")
                continue
            
            original_question = row['question']
            answer = row['answer']
            question_type = row['type']
            
            # Apply text augmentation
            augmented_question = augmentor.augment(original_question)
            
            # Store results
            result_entry = {
                'img_id': img_id,
                'img_path': img_path,
                'difficulty': difficulty,
                'original_question': original_question,
                'augmented_question': augmented_question,
                'answer': answer,
                'type': question_type,
                'is_changed': original_question != augmented_question
            }
            difficulty_results.append(result_entry)
            results.append(result_entry)
            
            # Display result
            status = "✓ CHANGED" if result_entry['is_changed'] else "○ NO CHANGE"
            print(f"\n{status} - Image: {img_id}")
            print(f"  Original:  {original_question}")
            print(f"  Augmented: {augmented_question}")
            print(f"  Answer: {answer}")
        
        # Save difficulty-specific results
        difficulty_output_path = os.path.join(output_dir, f"vivqa_augmented_{difficulty}.json")
        with open(difficulty_output_path, 'w', encoding='utf-8') as f:
            json.dump(difficulty_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Saved {difficulty} difficulty results to: {difficulty_output_path}")
    
    # Save combined results
    combined_output_path = os.path.join(output_dir, "vivqa_augmented_all.json")
    with open(combined_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved all results to: {combined_output_path}")
    
    # Generate statistics
    print(f"\n{'='*60}")
    print("Augmentation Statistics")
    print(f"{'='*60}")
    
    for difficulty, name in [(0.2, "easy"), (0.5, "medium"), (0.8, "hard")]:
        difficulty_samples = [r for r in results if r['difficulty'] == name]
        changed_count = sum(1 for r in difficulty_samples if r['is_changed'])
        total_count = len(difficulty_samples)
        change_rate = (changed_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\n{name.upper()}:")
        print(f"  Total samples: {total_count}")
        print(f"  Changed: {changed_count}")
        print(f"  Change rate: {change_rate:.1f}%")
    
    # Create a CSV summary
    csv_output_path = os.path.join(output_dir, "vivqa_augmented_summary.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_output_path, index=False, encoding='utf-8')
    print(f"\n✓ Saved CSV summary to: {csv_output_path}")
    
    # Display examples of different augmentation types
    print(f"\n{'='*60}")
    print("Examples of Augmentation Changes")
    print(f"{'='*60}")
    
    changed_results = [r for r in results if r['is_changed']]
    if changed_results:
        print("\nSuccessful augmentations:")
        for i, result in enumerate(changed_results[:5], 1):
            print(f"\n{i}. Image: {result['img_id']}")
            print(f"   Original:  {result['original_question']}")
            print(f"   Augmented: {result['augmented_question']}")
            print(f"   Difficulty: {result['difficulty']}")
    
    print(f"\n{'='*60}")
    print("ViVQA Augmentation Demonstration Complete!")
    print(f"{'='*60}")
    print(f"\nOutput files saved in: {output_dir}/")
    print("  - vivqa_augmented_easy.json")
    print("  - vivqa_augmented_medium.json")
    print("  - vivqa_augmented_hard.json")
    print("  - vivqa_augmented_all.json")
    print("  - vivqa_augmented_summary.csv")


def demonstrate_rule_based_augmentation():
    """Demonstrate rule-based text augmentation with Vietnamese linguistic rules."""
    print("="*60)
    print("Demonstrating Rule-Based Text Augmentation")
    print("="*60)
    
    # Sample Vietnamese questions from VQA datasets
    sample_questions = [
        "Có bao nhiêu người trong ảnh?",
        "Màu gì của chiếc xe này?",
        "Người đó đang làm gì?",
        "Đây có phải là con mèo không?",
        "Cái gì ở trên bàn?",
        "Bức ảnh này được chụp ở đâu?",
        "Con vật này có màu gì?",
        "Có phải trong ảnh có một chiếc ô tô đỏ?",
    ]
    
    print("\nOriginal questions and their augmentations:\n")
    
    for difficulty, name in [(0.2, "EASY"), (0.5, "MEDIUM"), (0.8, "HARD")]:
        print(f"\n{'-'*60}")
        print(f"Difficulty: {name} (difficulty={difficulty})")
        print(f"{'-'*60}")
        
        augmentor = RuleBasedTextAugmentation(difficulty=difficulty, seed=42)
        info = augmentor.get_augmentation_info()
        
        print(f"\nConfiguration:")
        print(f"  - Augmentation probability: {info['apply_prob']:.0%}")
        print(f"  - Max replacements per question: {info['max_replacements']}")
        print(f"  - Number of linguistic rules: {info['num_rules']}")
        
        print(f"\nSample augmentations:")
        for i, question in enumerate(sample_questions[:4], 1):
            # Generate multiple augmentations to show variety
            augmented = augmentor.augment(question)
            print(f"  {i}. Original:  {question}")
            print(f"     Augmented: {augmented}")
            if question != augmented:
                print(f"     ✓ Changed")
            else:
                print(f"     ○ Not changed (probability-based)")
            print()


def demonstrate_linguistic_rules():
    """Demonstrate different types of linguistic rules."""
    print("\n" + "="*60)
    print("Demonstrating Vietnamese Linguistic Rules")
    print("="*60)
    
    augmentor = RuleBasedTextAugmentation(difficulty=0.8, seed=42)
    
    # Examples for each rule category
    rule_examples = {
        "Question words": [
            "Cái gì ở trong hộp?",
            "Người này đang ở đâu?",
            "Sự kiện này diễn ra khi nào?",
        ],
        "Colors": [
            "Con vật có màu đỏ phải không?",
            "Chiếc áo có màu xanh hay vàng?",
        ],
        "Verbs": [
            "Người đó đang làm gì?",
            "Có phải họ đang ngồi không?",
        ],
        "Demonstratives": [
            "Cái này là gì?",
            "Người kia đang làm gì?",
        ],
        "Adjectives": [
            "Con vật này lớn hay nhỏ?",
            "Tòa nhà này cao không?",
        ],
    }
    
    print("\nRule-based transformations by category:")
    
    for category, questions in rule_examples.items():
        print(f"\n{category}:")
        for question in questions:
            # Generate multiple augmentations to show variations
            augmented_versions = []
            for _ in range(5):  # Try multiple times
                aug = augmentor.augment(question)
                if aug != question and aug not in augmented_versions:
                    augmented_versions.append(aug)
            
            print(f"  Original: {question}")
            if augmented_versions:
                for aug in augmented_versions[:3]:  # Show up to 3 variations
                    print(f"    → {aug}")
            else:
                print(f"    → (variations may appear with different random seeds)")
            print()


def demonstrate_curriculum_learning():
    """Demonstrate curriculum learning for text augmentation."""
    print("\n" + "="*60)
    print("Demonstrating Smooth Curriculum Learning Schedule")
    print("="*60)
    
    total_epochs = 30
    scheduler = CurriculumScheduler(
        total_epochs=total_epochs,
        strategy='cosine',
        warmup_ratio=0.1,
        difficulty_range=(0.1, 0.9)
    )
    
    print(f"\nTraining Schedule for {total_epochs} epochs:")
    print(f"  - Strategy: cosine")
    print(f"  - Warmup ratio: 10%")
    print(f"  - Difficulty range: 0.1 to 0.9")
    
    # Demonstrate text augmentation at different epochs
    sample_question = "Có bao nhiêu người trong ảnh này?"
    
    print(f"\nExample question: '{sample_question}'")
    print("\nAugmentation behavior across epochs:")
    
    for epoch in [0, 5, 9, 10, 15, 18, 19, 25, 29]:
        difficulty = scheduler.get_difficulty(epoch)
        augmentor = RuleBasedTextAugmentation(difficulty=difficulty, seed=42+epoch)
        
        # Try augmenting multiple times to show probability effect
        augmented_count = 0
        num_trials = 10
        for _ in range(num_trials):
            aug = augmentor.augment(sample_question)
            if aug != sample_question:
                augmented_count += 1
        
        info = augmentor.get_augmentation_info()
        actual_rate = (augmented_count / num_trials) * 100
        expected_rate = info['apply_prob'] * 100
        
        print(f"  Epoch {epoch:2d} ({difficulty}): "
              f"Expected {expected_rate:2.0f}% | Actual {actual_rate:2.0f}% augmented")


def demonstrate_factory_pattern():
    """Demonstrate using the factory pattern to create text augmentations."""
    print("\n" + "="*60)
    print("Demonstrating Factory Pattern for Text Augmentation")
    print("="*60)
    
    factory = AugmentationFactory()
    
    # List available augmentation types
    available_types = factory.get_available_text_augmentations()
    print(f"\nAvailable text augmentation types: {available_types}")
    
    # Create augmentation using factory
    print("\nCreating augmentations using factory:")
    
    augmentor = factory.create_text_augmentation(
        augmentation_type='rule-based',
        difficulty=0.5,
        seed=42
    )
    
    info = augmentor.get_augmentation_info()
    print(f"\nCreated: {info['type']}")
    print(f"  Difficulty: {info['difficulty']}")
    print(f"  Apply probability: {info['apply_prob']:.0%}")
    print(f"  Max replacements: {info['max_replacements']}")
    
    # Test with sample question
    question = "Người này đang làm gì ở trong ảnh?"
    augmented = augmentor.augment(question)
    print(f"\nTest augmentation:")
    print(f"  Original:  {question}")
    print(f"  Augmented: {augmented}")


def demonstrate_epoch_based_augmentation():
    """Demonstrate creating augmentors for different training epochs."""
    print("\n" + "="*60)
    print("Demonstrating Epoch-based Text Augmentation")
    print("="*60)
    
    total_epochs = 30
    scheduler = CurriculumScheduler(
        total_epochs=total_epochs,
        strategy='cosine',
        warmup_ratio=0.1
    )
    
    sample_questions = [
        "Có gì trong bức ảnh này?",
        "Người đó đang làm gì?",
        "Đây có phải là con chó không?",
    ]
    
    print("\nSimulating text augmentation across training epochs:")
    
    for epoch in [0, 9, 10, 18, 19, 29]:
        difficulty = scheduler.get_difficulty(epoch)
        augmentor = RuleBasedTextAugmentation(difficulty=difficulty, seed=42)
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch} - {difficulty} difficulty")
        print(f"{'='*50}")
        
        info = augmentor.get_augmentation_info()
        print(f"Apply probability: {info['apply_prob']:.0%}")
        print(f"Max replacements: {info['max_replacements']}")
        
        for question in sample_questions:
            augmented = augmentor.augment(question)
            status = "✓" if question != augmented else "○"
            print(f"\n{status} {question}")
            if question != augmented:
                print(f"  → {augmented}")


def demonstrate_batch_augmentation():
    """Demonstrate augmenting a batch of questions."""
    print("\n" + "="*60)
    print("Demonstrating Batch Text Augmentation")
    print("="*60)
    
    # Simulate a batch of questions from a VQA dataset
    question_batch = [
        "Có bao nhiêu người trong ảnh?",
        "Màu gì của chiếc xe này?",
        "Người đó đang làm gì?",
        "Đây có phải là con mèo không?",
        "Cái gì ở trên bàn?",
        "Bức ảnh này được chụp ở đâu?",
        "Con vật này có màu gì?",
        "Có phải trong ảnh có một chiếc ô tô đỏ?",
        "Thời tiết trong ảnh như thế nào?",
        "Người này đang ở đâu?",
    ]
    
    augmentor = RuleBasedTextAugmentation(difficulty=0.5, seed=42)
    
    print(f"\nAugmenting batch of {len(question_batch)} questions:")
    print(f"Difficulty: {augmentor.difficulty}")
    
    augmented_batch = []
    changed_count = 0
    
    for i, question in enumerate(question_batch, 1):
        augmented = augmentor.augment(question)
        augmented_batch.append(augmented)
        
        if augmented != question:
            changed_count += 1
            print(f"\n{i:2d}. ✓ Original:  {question}")
            print(f"      Augmented: {augmented}")
        else:
            print(f"\n{i:2d}. ○ {question} (not changed)")
    
    print(f"\nBatch statistics:")
    print(f"  Total questions: {len(question_batch)}")
    print(f"  Questions changed: {changed_count}")
    print(f"  Change rate: {(changed_count/len(question_batch)*100):.1f}%")


def demonstrate_integration_with_training():
    """Demonstrate how to integrate text augmentation in a training loop."""
    print("\n" + "="*60)
    print("Integration with Training Loop (Pseudo-code)")
    print("="*60)
    
    print("""
# Example integration in training pipeline:

from augmentation import RuleBasedTextAugmentation, CurriculumScheduler

# Initialize scheduler
total_epochs = 30
scheduler = CurriculumScheduler(
    total_epochs=total_epochs,
    strategy='cosine',
    warmup_ratio=0.1
)

# Training loop
for epoch in range(total_epochs):
    # Get difficulty level for current epoch
    difficulty = scheduler.get_difficulty(epoch)
    
    # Create text augmentor
    text_augmentor = RuleBasedTextAugmentation(
        difficulty=difficulty,
        seed=42 + epoch  # Different seed per epoch
    )
    
    # Training iteration
    for batch in train_dataloader:
        images, questions, answers = batch
        
        # Apply text augmentation
        augmented_questions = [
            text_augmentor.augment(q) for q in questions
        ]
        
        # Forward pass with augmented questions
        outputs = model(images, augmented_questions)
        loss = criterion(outputs, answers)
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}: {difficulty} augmentation")
    """)


def demonstrate_comparison_with_without_augmentation():
    """Compare questions with and without augmentation."""
    print("\n" + "="*60)
    print("Comparison: With vs Without Augmentation")
    print("="*60)
    
    questions = [
        "Có bao nhiêu người trong ảnh?",
        "Cái gì ở trên bàn?",
        "Người đó đang làm gì?",
        "Đây có phải là con mèo không?",
        "Bức ảnh này được chụp ở đâu?",
    ]
    
    augmentor_hard = RuleBasedTextAugmentation(difficulty=0.8, seed=42)
    
    print("\nGenerating multiple augmented versions (difficulty=0.8):")
    
    for question in questions:
        print(f"\nOriginal: {question}")
        print("Augmented versions:")
        
        # Generate multiple augmentations
        augmented_versions = set()
        for i in range(20):  # Try multiple times to get variations
            aug = augmentor_hard.augment(question)
            if aug != question:
                augmented_versions.add(aug)
        
        if augmented_versions:
            for j, aug in enumerate(list(augmented_versions)[:5], 1):
                print(f"  {j}. {aug}")
        else:
            print("  (No changes - may need different seed or more attempts)")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("TEXT AUGMENTATION WITH CURRICULUM LEARNING")
    print("Rule-Based Vietnamese Text Augmentation for VQA")
    print("="*60)
    
    # Run ViVQA augmentation demonstration first (with real data)
    demonstrate_vivqa_augmentation_with_images()
    
    # Run other demonstrations
    demonstrate_rule_based_augmentation()
    demonstrate_linguistic_rules()
    demonstrate_curriculum_learning()
    demonstrate_factory_pattern()
    demonstrate_epoch_based_augmentation()
    demonstrate_batch_augmentation()
    demonstrate_integration_with_training()
    demonstrate_comparison_with_without_augmentation()
    
    print("\n" + "="*60)
    print("All demonstrations completed successfully!")
    print("="*60)
    print("\nKey Takeaways:")
    print("  1. Real ViVQA dataset successfully augmented with text transformations")
    print("  2. Rule-based augmentation uses rich Vietnamese linguistic rules")
    print("  3. Curriculum learning controls augmentation FREQUENCY:")
    print("     - EASY: 20% of questions augmented")
    print("     - MEDIUM: 50% of questions augmented")
    print("     - HARD: 80% of questions augmented")
    print("  4. Multiple linguistic categories: question words, colors, verbs, etc.")
    print("  5. Factory pattern provides easy instantiation")
    print("  6. Seamless integration with training loops")
    print("\nNext Steps:")
    print("  - Review augmented results in runs/text_aug_demo/")
    print("  - Integrate with your VQA training pipeline")
    print("  - Experiment with different curriculum schedules")
    print("  - Combine with image augmentation for full data augmentation")
    print("  - Monitor model performance on augmented data")
    print("  - Add custom Vietnamese linguistic rules as needed")
    print()


if __name__ == "__main__":
    main()
