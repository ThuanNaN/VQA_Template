"""
Script to check the relationship between Curriculum Learning scheduler and
difficulty levels of textual and visual augmentation.

This script visualizes:
1. How difficulty progresses over epochs for different CL strategies
2. How augmentation phases change based on difficulty
3. Visual augmentation behavior (Easy: no aug, Medium: blur, Hard: mask)
4. Textual augmentation behavior (Easy: 2 paras, Medium: 1 para, Hard: 0 para)
"""
import os
import sys

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
from augmentation.scheduler import CurriculumScheduler
from augmentation.textual.rule_based import RuleBasedTextAugmentation
from augmentation.visual.mask import MaskedImageAugmentation


def get_augmentation_phase(difficulty: float, modality: str = 'visual') -> str:
    """
    Determine augmentation phase based on difficulty level.
    
    Args:
        difficulty: Difficulty value (0.0 - 1.0)
        modality: 'visual' or 'textual'
        
    Returns:
        Phase name and description
    """
    if difficulty < 0.33:
        if modality == 'visual':
            return 'Easy (No augmentation)'
        else:  # textual
            return 'Easy (2 paraphrases, 3 total)'
    elif difficulty < 0.66:
        if modality == 'visual':
            return 'Medium (Blur blocks)'
        else:  # textual
            return 'Medium (1 paraphrase, 2 total)'
    else:
        if modality == 'visual':
            return 'Hard (Mask blocks)'
        else:  # textual
            return 'Hard (No augmentation, 1 total)'


def visualize_cl_scheduler_relationship(total_epochs=100, save_path='cl_scheduler_analysis.png'):
    """
    Visualize the relationship between CL scheduler and augmentation difficulty.
    
    Args:
        total_epochs: Total number of training epochs
        save_path: Path to save the visualization
    """
    # Create different schedulers
    strategies = ['linear', 'cosine', 'exponential', 'step', 'polynomial']
    schedulers = {
        'linear': CurriculumScheduler(total_epochs, strategy='linear'),
        'cosine': CurriculumScheduler(total_epochs, strategy='cosine'),
        'exponential': CurriculumScheduler(total_epochs, strategy='exponential', gamma=0.1),
        'step': CurriculumScheduler(total_epochs, strategy='step', step_size=total_epochs//3),
        'polynomial': CurriculumScheduler(total_epochs, strategy='polynomial', power=2.0),
    }
    
    epochs = np.arange(total_epochs)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Difficulty progression for all strategies
    ax1 = plt.subplot(3, 2, 1)
    for name, scheduler in schedulers.items():
        difficulties = [scheduler.get_difficulty(ep) for ep in epochs]
        ax1.plot(epochs, difficulties, label=name.capitalize(), linewidth=2)
    
    # Add phase boundaries
    ax1.axhline(y=0.33, color='orange', linestyle='--', alpha=0.5, label='Easy/Medium boundary')
    ax1.axhline(y=0.66, color='red', linestyle='--', alpha=0.5, label='Medium/Hard boundary')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Difficulty', fontsize=12)
    ax1.set_title('Difficulty Progression Across Different CL Strategies', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # 2. Visual Augmentation Phases (using linear scheduler as example)
    ax2 = plt.subplot(3, 2, 2)
    scheduler = schedulers['linear']
    difficulties = [scheduler.get_difficulty(ep) for ep in epochs]
    
    # Color-code by phase
    colors = []
    for d in difficulties:
        if d < 0.33:
            colors.append('green')  # Easy - no aug
        elif d < 0.66:
            colors.append('orange')  # Medium - blur
        else:
            colors.append('red')  # Hard - mask
    
    ax2.scatter(epochs, difficulties, c=colors, alpha=0.6, s=20)
    ax2.axhspan(0, 0.33, alpha=0.2, color='green', label='Easy (No aug)')
    ax2.axhspan(0.33, 0.66, alpha=0.2, color='orange', label='Medium (Blur)')
    ax2.axhspan(0.66, 1.0, alpha=0.2, color='red', label='Hard (Mask)')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Difficulty', fontsize=12)
    ax2.set_title('Visual Augmentation Phases (Linear Scheduler)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    # 3. Textual Augmentation Phases (using linear scheduler)
    ax3 = plt.subplot(3, 2, 3)
    ax3.scatter(epochs, difficulties, c=colors, alpha=0.6, s=20)
    ax3.axhspan(0, 0.33, alpha=0.2, color='green', label='Easy (2 paras, 3 total)')
    ax3.axhspan(0.33, 0.66, alpha=0.2, color='orange', label='Medium (1 para, 2 total)')
    ax3.axhspan(0.66, 1.0, alpha=0.2, color='red', label='Hard (0 para, 1 total)')
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Difficulty', fontsize=12)
    ax3.set_title('Textual Augmentation Phases (Linear Scheduler)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.05, 1.05)
    
    # 4. Number of outputs over epochs for textual augmentation
    ax4 = plt.subplot(3, 2, 4)
    num_outputs = []
    for d in difficulties:
        if d < 0.33:
            num_outputs.append(3)  # original + 2 paraphrases
        elif d < 0.66:
            num_outputs.append(2)  # original + 1 paraphrase
        else:
            num_outputs.append(1)  # only original
    
    ax4.plot(epochs, num_outputs, linewidth=2, color='blue')
    ax4.fill_between(epochs, num_outputs, alpha=0.3, color='blue')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Number of Text Outputs', fontsize=12)
    ax4.set_title('Textual Augmentation Output Count Over Training', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 4)
    
    # 5. Phase duration comparison
    ax5 = plt.subplot(3, 2, 5)
    
    phase_data = {}
    for name, scheduler in schedulers.items():
        difficulties = [scheduler.get_difficulty(ep) for ep in epochs]
        easy_count = sum(1 for d in difficulties if d < 0.33)
        medium_count = sum(1 for d in difficulties if 0.33 <= d < 0.66)
        hard_count = sum(1 for d in difficulties if d >= 0.66)
        phase_data[name] = [easy_count, medium_count, hard_count]
    
    x = np.arange(len(strategies))
    width = 0.25
    
    ax5.bar(x - width, [phase_data[s][0] for s in strategies], width, label='Easy', color='green', alpha=0.7)
    ax5.bar(x, [phase_data[s][1] for s in strategies], width, label='Medium', color='orange', alpha=0.7)
    ax5.bar(x + width, [phase_data[s][2] for s in strategies], width, label='Hard', color='red', alpha=0.7)
    
    ax5.set_xlabel('CL Strategy', fontsize=12)
    ax5.set_ylabel('Number of Epochs', fontsize=12)
    ax5.set_title('Phase Duration by CL Strategy', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([s.capitalize() for s in strategies])
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Comparison of strategies at key epochs
    ax6 = plt.subplot(3, 2, 6)
    
    key_epochs = [0, total_epochs//4, total_epochs//2, 3*total_epochs//4, total_epochs-1]
    key_epoch_labels = ['Start', 'Q1', 'Mid', 'Q3', 'End']
    
    for name, scheduler in schedulers.items():
        difficulties_at_key = [scheduler.get_difficulty(ep) for ep in key_epochs]
        ax6.plot(key_epoch_labels, difficulties_at_key, marker='o', label=name.capitalize(), linewidth=2, markersize=8)
    
    ax6.axhline(y=0.33, color='orange', linestyle='--', alpha=0.5)
    ax6.axhline(y=0.66, color='red', linestyle='--', alpha=0.5)
    
    ax6.set_xlabel('Training Progress', fontsize=12)
    ax6.set_ylabel('Difficulty', fontsize=12)
    ax6.set_title('Difficulty at Key Training Milestones', fontsize=14, fontweight='bold')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")
    plt.close()


def print_detailed_analysis(total_epochs=100):
    """
    Print detailed analysis of CL scheduler and augmentation relationship.
    
    Args:
        total_epochs: Total number of training epochs
    """
    print("=" * 80)
    print("CURRICULUM LEARNING SCHEDULER & AUGMENTATION DIFFICULTY ANALYSIS")
    print("=" * 80)
    
    # Analyze different strategies
    strategies = {
        'linear': CurriculumScheduler(total_epochs, strategy='linear'),
        'cosine': CurriculumScheduler(total_epochs, strategy='cosine'),
        'exponential': CurriculumScheduler(total_epochs, strategy='exponential', gamma=0.1),
        'step': CurriculumScheduler(total_epochs, strategy='step', step_size=total_epochs//3),
        'polynomial': CurriculumScheduler(total_epochs, strategy='polynomial', power=2.0),
    }
    
    for name, scheduler in strategies.items():
        print(f"\n{'-' * 80}")
        print(f"Strategy: {name.upper()}")
        print(f"{'-' * 80}")
        
        # Get schedule info
        info = scheduler.get_schedule_info()
        print(f"\nConfiguration:")
        print(f"  Total Epochs: {info['total_epochs']}")
        print(f"  Difficulty Range: [{info['min_difficulty']:.2f}, {info['max_difficulty']:.2f}]")
        print(f"  Warmup Epochs: {info['warmup_epochs']}")
        
        # Show difficulties at key epochs
        print(f"\nDifficulty Progression:")
        key_epochs = [0, total_epochs//4, total_epochs//2, 3*total_epochs//4, total_epochs-1]
        
        for epoch in key_epochs:
            difficulty = scheduler.get_difficulty(epoch)
            visual_phase = get_augmentation_phase(difficulty, 'visual')
            textual_phase = get_augmentation_phase(difficulty, 'textual')
            
            print(f"  Epoch {epoch:3d}: difficulty={difficulty:.4f}")
            print(f"    → Visual:  {visual_phase}")
            print(f"    → Textual: {textual_phase}")
        
        # Count epochs in each phase
        difficulties = [scheduler.get_difficulty(ep) for ep in range(total_epochs)]
        easy_epochs = sum(1 for d in difficulties if d < 0.33)
        medium_epochs = sum(1 for d in difficulties if 0.33 <= d < 0.66)
        hard_epochs = sum(1 for d in difficulties if d >= 0.66)
        
        print(f"\nPhase Distribution:")
        print(f"  Easy:   {easy_epochs:3d} epochs ({easy_epochs/total_epochs*100:5.1f}%)")
        print(f"  Medium: {medium_epochs:3d} epochs ({medium_epochs/total_epochs*100:5.1f}%)")
        print(f"  Hard:   {hard_epochs:3d} epochs ({hard_epochs/total_epochs*100:5.1f}%)")
    
    # Show augmentation behavior examples
    print(f"\n{'=' * 80}")
    print("AUGMENTATION BEHAVIOR BY PHASE")
    print(f"{'=' * 80}")
    
    print("\nVISUAL AUGMENTATION (Image):")
    print("-" * 80)
    for difficulty, phase_name in [(0.0, "Easy"), (0.5, "Medium"), (1.0, "Hard")]:
        print(f"\n{phase_name} Phase (difficulty={difficulty:.1f}):")
        visual_aug = MaskedImageAugmentation(difficulty=difficulty, seed=42)
        print(f"  Phase: {visual_aug.augmentation_phase}")
        print(f"  Blur blocks: {visual_aug.apply_blur_blocks}")
        print(f"  Mask blocks: {visual_aug.apply_mask_blocks}")
        print(f"  Mask ratio: {visual_aug.mask_ratio:.4f}")
    
    print("\n" + "-" * 80)
    print("TEXTUAL AUGMENTATION (Question):")
    print("-" * 80)
    for difficulty, phase_name in [(0.0, "Easy"), (0.5, "Medium"), (1.0, "Hard")]:
        print(f"\n{phase_name} Phase (difficulty={difficulty:.1f}):")
        text_aug = RuleBasedTextAugmentation(difficulty=difficulty, seed=42)
        info = text_aug.get_augmentation_info()
        print(f"  Phase: {info['augmentation_phase']}")
        print(f"  Paraphrases generated: {info['num_paraphrases']}")
        print(f"  Total outputs: {info['total_outputs']} (original + paraphrases)")
    
    print(f"\n{'=' * 80}")
    print("3-PHASE STRATEGY SUMMARY")
    print(f"{'=' * 80}")
    print("\n┌─────────────┬──────────────────┬────────────────────────┬─────────────────────────┐")
    print("│ Phase       │ Difficulty Range │ Visual Augmentation    │ Textual Augmentation    │")
    print("├─────────────┼──────────────────┼────────────────────────┼─────────────────────────┤")
    print("│ Easy        │ 0.0 - 0.33       │ No augmentation        │ 2 paraphrases (3 total) │")
    print("│ Medium      │ 0.33 - 0.66      │ Blur random blocks     │ 1 paraphrase (2 total)  │")
    print("│ Hard        │ 0.66 - 1.0       │ Mask random blocks     │ No augmentation (1 tot) │")
    print("└─────────────┴──────────────────┴────────────────────────┴─────────────────────────┘")
    
    print(f"\n{'=' * 80}\n")


def test_augmentation_at_difficulty_levels():
    """Test actual augmentation behavior at different difficulty levels."""
    
    print("=" * 80)
    print("LIVE AUGMENTATION TEST")
    print("=" * 80)
    
    test_text = "Cô giáo giải thích bài rất rõ ràng"
    
    print(f"\nTest sentence: {test_text}")
    print("\n" + "-" * 80)
    
    difficulties = [
        (0.0, "Easy"),
        (0.3, "Easy (boundary)"),
        (0.5, "Medium"),
        (0.65, "Medium (boundary)"),
        (1.0, "Hard")
    ]
    
    for difficulty, label in difficulties:
        print(f"\nDifficulty = {difficulty:.2f} ({label}):")
        
        text_aug = RuleBasedTextAugmentation(difficulty=difficulty, seed=42)
        outputs = text_aug.augment(test_text)
        
        print(f"  Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs, 1):
            marker = "[original]" if i == 1 else "[paraphrase]"
            print(f"    {i}. {output} {marker}")
    
    print("\n" + "=" * 80 + "\n")


def main():
    """Main function to run all analyses."""
    
    print("\n" + "=" * 80)
    print(" CURRICULUM LEARNING SCHEDULER & AUGMENTATION ANALYSIS SCRIPT")
    print("=" * 80 + "\n")
    
    # Configuration
    total_epochs = 30
    
    # 1. Print detailed analysis
    print_detailed_analysis(total_epochs)
    
    # 2. Test live augmentation
    test_augmentation_at_difficulty_levels()
    
    # 3. Create visualization
    print("Generating visualization...")
    visualize_cl_scheduler_relationship(total_epochs)
    
    print("\n" + "=" * 80)
    print(" Analysis Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - cl_scheduler_analysis.png (visualization)")
    print("\nKey Insights:")
    print("  • Visual & Textual augmentation use the same 3-phase strategy")
    print("  • Easy phase: Most augmentation for textual, none for visual")
    print("  • Hard phase: Most augmentation for visual, none for textual")
    print("  • Different CL strategies provide different phase distributions")
    print("  • Linear strategy gives equal time in each phase")
    print("  • Cosine strategy provides smooth transitions")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
