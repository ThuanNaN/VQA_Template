"""
Integration Example: Using Vietnamese Augmentation with ViVQA Dataset

This example shows how to integrate the rule-based augmentation
with the existing ViVQA dataset for training.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pandas as pd

# Import directly from augmentation module to avoid torch dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "augmentation",
    os.path.join(parent_dir, "utils", "augmentation.py")
)
augmentation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(augmentation)

VietnameseVQAAugmentation = augmentation.VietnameseVQAAugmentation
create_augmented_dataset = augmentation.create_augmented_dataset


def augment_vivqa_csv(
    input_csv: str,
    output_csv: str,
    num_augmentations: int = 1,
    seed: int = 42
):
    """
    Augment a ViVQA CSV dataset file.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to save augmented CSV file
        num_augmentations: Number of augmented versions per question
        seed: Random seed for reproducibility
    """
    print(f"Loading dataset from: {input_csv}")
    
    # Load original data
    df = pd.read_csv(input_csv)
    original_data = df.to_dict('records')
    
    print(f"Original dataset size: {len(original_data)}")
    
    # Create augmented dataset
    print(f"Generating {num_augmentations} augmentation(s) per question...")
    augmented_data = create_augmented_dataset(
        original_data,
        num_augmentations=num_augmentations,
        seed=seed
    )
    
    print(f"Augmented dataset size: {len(augmented_data)}")
    print(f"New samples added: {len(augmented_data) - len(original_data)}")
    
    # Convert to DataFrame and save
    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv(output_csv, index=False)
    
    print(f"Saved augmented dataset to: {output_csv}")
    
    # Show some examples
    print("\n" + "="*70)
    print("Sample Augmented Questions:")
    print("="*70)
    
    augmented_only = [item for item in augmented_data if item.get('augmented', False)]
    for i, item in enumerate(augmented_only[:5], 1):
        print(f"\n{i}. Question: {item['question']}")
        print(f"   Answer: {item['answer']}")
        print(f"   Image ID: {item['img_id']}")


def analyze_augmentation_potential(csv_path: str):
    """
    Analyze how many questions in a dataset can be augmented.
    
    Args:
        csv_path: Path to CSV file
    """
    print("="*70)
    print("Augmentation Potential Analysis")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    questions = df['question'].tolist()
    
    augmentor = VietnameseVQAAugmentation(seed=42)
    stats = augmentor.get_statistics(questions)
    
    print(f"\nDataset: {csv_path}")
    print(f"Total questions: {stats['total_questions']}")
    print(f"Questions with question words: {stats['questions_with_question_words']} ({stats['questions_with_question_words']/stats['total_questions']*100:.1f}%)")
    print(f"Questions with synonyms: {stats['questions_with_synonyms']} ({stats['questions_with_synonyms']/stats['total_questions']*100:.1f}%)")
    print(f"Questions with demonstratives: {stats['questions_with_demonstratives']} ({stats['questions_with_demonstratives']/stats['total_questions']*100:.1f}%)")
    print(f"Total replaceable words: {stats['total_replaceable_words']}")
    print(f"Average replaceable words per question: {stats['total_replaceable_words']/stats['total_questions']:.2f}")
    
    # Test augmentation on a sample
    print("\n" + "="*70)
    print("Sample Augmentations:")
    print("="*70)
    
    sample_questions = questions[:5]
    for i, question in enumerate(sample_questions, 1):
        print(f"\n{i}. Original: {question}")
        augmented = augmentor.augment_question(question, num_augmentations=2)
        for j, aug_q in enumerate(augmented, 1):
            print(f"   Aug {j}: {aug_q}")


def create_training_ready_dataset(
    train_csv: str,
    output_csv: str,
    augmentation_ratio: float = 0.5,
    seed: int = 42
):
    """
    Create a training-ready dataset with controlled augmentation.
    
    Args:
        train_csv: Path to original training CSV
        output_csv: Path to save augmented training CSV
        augmentation_ratio: Ratio of augmented to original samples (0.5 = 50% augmented)
        seed: Random seed
    """
    print("="*70)
    print("Creating Training-Ready Dataset")
    print("="*70)
    
    df = pd.read_csv(train_csv)
    original_size = len(df)
    
    # Calculate number of augmentations needed
    target_augmented = int(original_size * augmentation_ratio)
    num_augmentations = max(1, target_augmented // original_size)
    
    print(f"\nOriginal size: {original_size}")
    print(f"Augmentation ratio: {augmentation_ratio}")
    print(f"Augmentations per question: {num_augmentations}")
    
    # Create augmented dataset
    original_data = df.to_dict('records')
    augmented_data = create_augmented_dataset(
        original_data,
        num_augmentations=num_augmentations,
        seed=seed
    )
    
    # If we have too many, randomly sample
    if len(augmented_data) > original_size * (1 + augmentation_ratio):
        import random
        random.seed(seed)
        
        # Separate original and augmented
        original_items = [item for item in augmented_data if not item.get('augmented', False)]
        augmented_items = [item for item in augmented_data if item.get('augmented', False)]
        
        # Sample the right number of augmented items
        target_count = int(original_size * augmentation_ratio)
        sampled_augmented = random.sample(augmented_items, min(target_count, len(augmented_items)))
        
        # Combine
        augmented_data = original_items + sampled_augmented
    
    # Save
    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv(output_csv, index=False)
    
    print(f"Final dataset size: {len(augmented_data)}")
    print(f"Actual augmentation ratio: {(len(augmented_data) - original_size) / original_size:.2f}")
    print(f"Saved to: {output_csv}")


def main():
    """Demonstrate integration examples."""
    
    print("\n" + "="*70)
    print("VIETNAMESE AUGMENTATION - DATASET INTEGRATION EXAMPLES")
    print("="*70 + "\n")
    
    # Example 1: Analyze potential (works without actual data)
    print("\nExample 1: Analyzing augmentation potential")
    print("-" * 70)
    print("This would analyze a real dataset if available:")
    print("""
    analyze_augmentation_potential('data/vivqa/train.csv')
    """)
    
    # Example 2: Simple augmentation
    print("\nExample 2: Simple dataset augmentation")
    print("-" * 70)
    print("Usage:")
    print("""
    augment_vivqa_csv(
        input_csv='data/vivqa/train.csv',
        output_csv='data/vivqa/train_augmented.csv',
        num_augmentations=1,
        seed=42
    )
    """)
    
    # Example 3: Controlled augmentation
    print("\nExample 3: Controlled augmentation ratio")
    print("-" * 70)
    print("Usage:")
    print("""
    create_training_ready_dataset(
        train_csv='data/vivqa/train.csv',
        output_csv='data/vivqa/train_aug_50pct.csv',
        augmentation_ratio=0.5,  # Add 50% more samples via augmentation
        seed=42
    )
    """)
    
    # Example 4: In-memory augmentation
    print("\nExample 4: In-memory augmentation for custom workflows")
    print("-" * 70)
    print("Usage:")
    print("""
    from utils import VietnameseVQAAugmentation
    
    augmentor = VietnameseVQAAugmentation(seed=42)
    
    # During data loading
    for idx, row in df.iterrows():
        original_question = row['question']
        
        # Generate augmented versions
        augmented_questions = augmentor.augment_question(
            original_question, 
            num_augmentations=2
        )
        
        # Use in your training loop
        for aug_q in augmented_questions:
            # Process augmented question with same answer and image
            process_sample(aug_q, row['answer'], row['img_id'])
    """)
    
    # Demonstrate with sample data
    print("\n" + "="*70)
    print("DEMONSTRATION WITH SAMPLE DATA")
    print("="*70)
    
    # Create sample dataset
    sample_data = [
        {"question": "Cái gì trong ảnh này?", "answer": "con mèo", "img_id": "000001"},
        {"question": "Người này đang làm gì?", "answer": "đọc sách", "img_id": "000002"},
        {"question": "Màu của chiếc xe là gì?", "answer": "đỏ", "img_id": "000003"},
        {"question": "Có bao nhiêu con chó?", "answer": "hai", "img_id": "000004"},
        {"question": "Thời tiết như thế nào?", "answer": "nắng", "img_id": "000005"},
    ]
    
    print("\nCreating sample CSV...")
    sample_df = pd.DataFrame(sample_data)
    sample_csv = "/tmp/sample_vivqa.csv"
    sample_df.to_csv(sample_csv, index=False)
    
    print("\nAugmenting sample dataset...")
    augment_vivqa_csv(
        input_csv=sample_csv,
        output_csv="/tmp/sample_vivqa_augmented.csv",
        num_augmentations=2,
        seed=42
    )
    
    print("\n" + "="*70)
    print("INTEGRATION EXAMPLES COMPLETE")
    print("="*70)
    print("\nTo use with your actual ViVQA dataset:")
    print("1. Ensure your dataset is downloaded (see README.md)")
    print("2. Run the functions above with your actual data paths")
    print("3. Use the augmented CSV in your training pipeline")


if __name__ == "__main__":
    main()
