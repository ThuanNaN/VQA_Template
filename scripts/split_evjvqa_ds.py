import json
import random
from pathlib import Path

def split_evjvqa_dataset(input_file, output_dir, train_ratio=0.7, val_ratio=0.3, test_ratio=0.1, seed=42):
    """
    Split EVJVQA dataset into train, val, and test sets.
    
    Args:
        input_file: Path to evjvqa_train.json
        output_dir: Directory to save split files
        train_ratio: Ratio for training set (default: 0.7)
        val_ratio: Ratio for validation set (default: 0.3)
        test_ratio: Ratio for test set (default: 0.1)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load the original dataset
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    
    print(f"Total images: {len(images)}")
    print(f"Total annotations: {len(annotations)}")
    
    # Normalize ratios to sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio = train_ratio / total_ratio
    val_ratio = val_ratio / total_ratio
    test_ratio = test_ratio / total_ratio
    
    # Shuffle images
    random.shuffle(images)
    
    # Calculate split indices
    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    
    # Split images
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    print(f"\nSplit statistics:")
    print(f"Train images: {len(train_images)} ({len(train_images)/total_images*100:.1f}%)")
    print(f"Val images: {len(val_images)} ({len(val_images)/total_images*100:.1f}%)")
    print(f"Test images: {len(test_images)} ({len(test_images)/total_images*100:.1f}%)")
    
    # Create image ID sets for each split
    train_image_ids = set(img['id'] for img in train_images)
    val_image_ids = set(img['id'] for img in val_images)
    test_image_ids = set(img['id'] for img in test_images)
    
    # Split annotations based on image IDs
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in annotations if ann['image_id'] in val_image_ids]
    test_annotations = [ann for ann in annotations if ann['image_id'] in test_image_ids]
    
    print(f"\nAnnotation statistics:")
    print(f"Train annotations: {len(train_annotations)}")
    print(f"Val annotations: {len(val_annotations)}")
    print(f"Test annotations: {len(test_annotations)}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save split datasets
    splits = {
        'train': {'images': train_images, 'annotations': train_annotations},
        'val': {'images': val_images, 'annotations': val_annotations},
        'test': {'images': test_images, 'annotations': test_annotations}
    }
    
    for split_name, split_data in splits.items():
        output_file = output_dir / f'evjvqa_{split_name}.json'
        print(f"\nSaving {split_name} set to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {len(split_data['images'])} images and {len(split_data['annotations'])} annotations")
    
    print("\n✓ Dataset split completed successfully!")
    return splits


if __name__ == "__main__":
    # Configuration
    input_file = "data/evjvqa/evjvqa_train.json"
    output_dir = "data/evjvqa"
    
    # Split with ratio 7:3:1 (train:val:test)
    split_evjvqa_dataset(
        input_file=input_file,
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.3,
        test_ratio=0.1,
        seed=42
    )
