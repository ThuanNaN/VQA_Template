#!/bin/bash
# Example: Training with Wrong Prediction Tracking
# This script demonstrates how to use the wrong prediction tracking feature

echo "================================"
echo "Wrong Prediction Tracking Demo"
echo "================================"
echo ""

# Example 1: Basic training with default wrong prediction tracking
echo "Example 1: Basic training (wrong prediction tracking enabled by default)"
echo "Command:"
echo "  python train.py \\"
echo "    --dataset_name vivqa \\"
echo "    --epochs 5 \\"
echo "    --batch_size 32"
echo ""
echo "Output will be saved to: wrong_predictions/vivqa-run-71/epoch_XXX/"
echo ""

# Example 2: Custom output directory
echo "Example 2: Custom output directory"
echo "Command:"
echo "  python train.py \\"
echo "    --dataset_name vivqa \\"
echo "    --epochs 5 \\"
echo "    --wrong_prediction_dir my_wrong_predictions"
echo ""
echo "Output will be saved to: my_wrong_predictions/vivqa-run-71/epoch_XXX/"
echo ""

# Example 3: Disable wrong prediction tracking
echo "Example 3: Disable wrong prediction tracking"
echo "Command:"
echo "  python train.py \\"
echo "    --dataset_name vivqa \\"
echo "    --epochs 5 \\"
echo "    --disable_wrong_prediction_tracking"
echo ""
echo "No wrong predictions will be tracked"
echo ""

# Example 4: Full experiment with all features
echo "Example 4: Full experiment with all tracking features"
echo "Command:"
echo "  python train.py \\"
echo "    --dataset_name vivqa \\"
echo "    --epochs 30 \\"
echo "    --enable_image_augmentation \\"
echo "    --enable_text_augmentation \\"
echo "    --enable_curriculum \\"
echo "    --enable_sample_observation \\"
echo "    --enable_wrong_prediction_tracking \\"
echo "    --run_name full_experiment"
echo ""
echo "This will enable:"
echo "  - Image augmentation"
echo "  - Text augmentation"
echo "  - Curriculum learning"
echo "  - Sample observation"
echo "  - Wrong prediction tracking"
echo ""

echo "================================"
echo "Analyzing Wrong Predictions"
echo "================================"
echo ""

# Example analysis script
cat > /tmp/analyze_wrong_predictions.py << 'EOF'
#!/usr/bin/env python3
"""
Example: Analyze wrong predictions from training
"""
import json
from pathlib import Path
from collections import Counter

def analyze_wrong_predictions(base_dir):
    """Analyze wrong predictions across epochs."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    # Get all epoch directories
    epoch_dirs = sorted(base_path.glob('epoch_*'))
    
    if not epoch_dirs:
        print("No epoch data found")
        return
    
    print(f"Found {len(epoch_dirs)} epochs of data")
    print()
    
    # Track metrics
    epoch_error_counts = []
    all_wrong_indices = set()
    wrong_by_epoch = {}
    question_types = Counter()
    
    for epoch_dir in epoch_dirs:
        json_file = epoch_dir / 'wrong_predictions.json'
        
        if not json_file.exists():
            continue
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        epoch = data['epoch']
        num_wrong = data['num_wrong_predictions']
        
        epoch_error_counts.append((epoch, num_wrong))
        wrong_by_epoch[epoch] = set()
        
        # Collect indices and analyze questions
        for sample in data['wrong_samples']:
            idx = sample['sample_idx']
            all_wrong_indices.add(idx)
            wrong_by_epoch[epoch].add(idx)
            
            # Simple question type classification
            question = sample['question'].lower()
            if 'màu' in question or 'color' in question:
                question_types['color'] += 1
            elif 'bao nhiêu' in question or 'how many' in question:
                question_types['counting'] += 1
            elif 'ai' in question or 'who' in question:
                question_types['person'] += 1
            elif 'đâu' in question or 'where' in question:
                question_types['location'] += 1
            elif 'gì' in question or 'what' in question:
                question_types['object'] += 1
            else:
                question_types['other'] += 1
    
    # Print analysis
    print("=" * 60)
    print("Error Count by Epoch")
    print("=" * 60)
    for epoch, count in epoch_error_counts:
        print(f"  Epoch {epoch:3d}: {count:4d} wrong predictions")
    print()
    
    # Find persistent errors
    print("=" * 60)
    print("Persistent Errors Analysis")
    print("=" * 60)
    persistent_errors = {}
    for idx in all_wrong_indices:
        count = sum(1 for wrong_set in wrong_by_epoch.values() if idx in wrong_set)
        if count >= 3:  # Wrong in at least 3 epochs
            persistent_errors[idx] = count
    
    if persistent_errors:
        print(f"Found {len(persistent_errors)} samples wrong in 3+ epochs:")
        sorted_errors = sorted(persistent_errors.items(), key=lambda x: x[1], reverse=True)
        for idx, count in sorted_errors[:10]:  # Top 10
            print(f"  Sample {idx}: wrong in {count} epochs")
    else:
        print("No persistent errors found (samples wrong in 3+ epochs)")
    print()
    
    # Question type analysis
    print("=" * 60)
    print("Error Distribution by Question Type")
    print("=" * 60)
    total_errors = sum(question_types.values())
    for qtype, count in question_types.most_common():
        percentage = (count / total_errors * 100) if total_errors > 0 else 0
        print(f"  {qtype.capitalize():12s}: {count:4d} ({percentage:5.1f}%)")
    print()
    
    # Improvement analysis
    if len(epoch_error_counts) >= 2:
        print("=" * 60)
        print("Improvement Analysis")
        print("=" * 60)
        first_epoch, first_count = epoch_error_counts[0]
        last_epoch, last_count = epoch_error_counts[-1]
        
        improvement = first_count - last_count
        improvement_pct = (improvement / first_count * 100) if first_count > 0 else 0
        
        print(f"  First epoch ({first_epoch}): {first_count} errors")
        print(f"  Last epoch ({last_epoch}): {last_count} errors")
        print(f"  Improvement: {improvement} errors ({improvement_pct:+.1f}%)")
        print()

# Example usage
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        # Default directory
        base_dir = 'wrong_predictions/vivqa-run-71'
    
    print(f"Analyzing wrong predictions in: {base_dir}")
    print()
    analyze_wrong_predictions(base_dir)
EOF

chmod +x /tmp/analyze_wrong_predictions.py

echo "Created analysis script: /tmp/analyze_wrong_predictions.py"
echo ""
echo "Usage:"
echo "  python /tmp/analyze_wrong_predictions.py wrong_predictions/vivqa-run-71"
echo ""
echo "This script will:"
echo "  - Show error counts by epoch"
echo "  - Identify persistent errors (samples wrong in multiple epochs)"
echo "  - Analyze error distribution by question type"
echo "  - Calculate improvement over training"
echo ""

echo "================================"
echo "For more information, see:"
echo "  docs/WRONG_PREDICTION_TRACKING.md"
echo "================================"
