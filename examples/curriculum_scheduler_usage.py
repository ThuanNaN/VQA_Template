"""
Example usage of Curriculum Learning Schedulers.

This script demonstrates how to use both the discrete CurriculumLearningScheduler
and the new SmoothCurriculumScheduler for fine-grained difficulty control.
"""

import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
from augmentation.scheduler import CurriculumLearningScheduler, SmoothCurriculumScheduler


def plot_scheduler_comparison():
    """Compare different scheduling strategies visually."""
    total_epochs = 100
    epochs = list(range(total_epochs))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Curriculum Learning Scheduler Comparison', fontsize=16)
    
    # 1. Discrete Scheduler (Original)
    discrete_scheduler = CurriculumLearningScheduler(total_epochs=total_epochs)
    discrete_difficulties = []
    for epoch in epochs:
        level = discrete_scheduler.get_difficulty_for_epoch(epoch)
        # Map to numeric values for plotting
        level_map = {'easy': 0.0, 'medium': 0.5, 'hard': 1.0}
        discrete_difficulties.append(level_map[level.value])
    
    axes[0, 0].plot(epochs, discrete_difficulties, 'b-', linewidth=2)
    axes[0, 0].set_title('Discrete Levels (Original)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Difficulty')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.1, 1.1)
    
    # 2. Linear Smooth Scheduler
    linear_scheduler = SmoothCurriculumScheduler(
        total_epochs=total_epochs, 
        strategy='linear'
    )
    linear_difficulties = [linear_scheduler.get_difficulty(epoch) for epoch in epochs]
    
    axes[0, 1].plot(epochs, linear_difficulties, 'g-', linewidth=2)
    axes[0, 1].set_title('Linear Progression')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Difficulty')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-0.1, 1.1)
    
    # 3. Cosine Smooth Scheduler
    cosine_scheduler = SmoothCurriculumScheduler(
        total_epochs=total_epochs, 
        strategy='cosine'
    )
    cosine_difficulties = [cosine_scheduler.get_difficulty(epoch) for epoch in epochs]
    
    axes[0, 2].plot(epochs, cosine_difficulties, 'r-', linewidth=2)
    axes[0, 2].set_title('Cosine Annealing')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Difficulty')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(-0.1, 1.1)
    
    # 4. Exponential Smooth Scheduler
    exp_scheduler = SmoothCurriculumScheduler(
        total_epochs=total_epochs, 
        strategy='exponential',
        gamma=0.05
    )
    exp_difficulties = [exp_scheduler.get_difficulty(epoch) for epoch in epochs]
    
    axes[1, 0].plot(epochs, exp_difficulties, 'm-', linewidth=2)
    axes[1, 0].set_title('Exponential Growth (γ=0.05)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Difficulty')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-0.1, 1.1)
    
    # 5. Step Smooth Scheduler
    step_scheduler = SmoothCurriculumScheduler(
        total_epochs=total_epochs, 
        strategy='step',
        step_size=20,
        gamma=0.25
    )
    step_difficulties = [step_scheduler.get_difficulty(epoch) for epoch in epochs]
    
    axes[1, 1].plot(epochs, step_difficulties, 'c-', linewidth=2)
    axes[1, 1].set_title('Step-wise (step=20, γ=0.25)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Difficulty')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(-0.1, 1.1)
    
    # 6. Polynomial Smooth Scheduler
    poly_scheduler = SmoothCurriculumScheduler(
        total_epochs=total_epochs, 
        strategy='polynomial',
        power=2.5
    )
    poly_difficulties = [poly_scheduler.get_difficulty(epoch) for epoch in epochs]
    
    axes[1, 2].plot(epochs, poly_difficulties, 'y-', linewidth=2)
    axes[1, 2].set_title('Polynomial (power=2.5)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Difficulty')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('runs/curriculum_schedulers_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved comparison plot to 'runs/curriculum_schedulers_comparison.png'")
    plt.show()


def demonstrate_warmup():
    """Demonstrate warmup functionality."""
    total_epochs = 100
    warmup_epochs = 20
    
    scheduler = SmoothCurriculumScheduler(
        total_epochs=total_epochs,
        strategy='cosine',
        warmup_epochs=warmup_epochs
    )
    
    epochs = list(range(total_epochs))
    difficulties = [scheduler.get_difficulty(epoch) for epoch in epochs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, difficulties, 'b-', linewidth=2, label='Cosine with Warmup')
    plt.axvline(x=warmup_epochs, color='r', linestyle='--', 
                label=f'Warmup End (epoch {warmup_epochs})')
    plt.axhspan(0, 0.05, alpha=0.2, color='red', label='Warmup Phase')
    
    plt.title('Curriculum Scheduler with Warmup')
    plt.xlabel('Epoch')
    plt.ylabel('Difficulty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('runs/curriculum_scheduler_warmup.png', dpi=150, bbox_inches='tight')
    print("✓ Saved warmup plot to 'runs/curriculum_scheduler_warmup.png'")
    plt.show()


def demonstrate_custom_range():
    """Demonstrate custom difficulty ranges."""
    total_epochs = 100
    
    # Custom range: start at 0.2, end at 0.8
    scheduler = SmoothCurriculumScheduler(
        total_epochs=total_epochs,
        strategy='linear',
        min_difficulty=0.2,
        max_difficulty=0.8
    )
    
    epochs = list(range(total_epochs))
    difficulties = [scheduler.get_difficulty(epoch) for epoch in epochs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, difficulties, 'g-', linewidth=2)
    plt.axhline(y=0.2, color='b', linestyle='--', alpha=0.5, label='Min Difficulty (0.2)')
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Max Difficulty (0.8)')
    
    plt.title('Custom Difficulty Range (0.2 to 0.8)')
    plt.xlabel('Epoch')
    plt.ylabel('Difficulty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.savefig('runs/curriculum_scheduler_custom_range.png', dpi=150, bbox_inches='tight')
    print("✓ Saved custom range plot to 'runs/curriculum_scheduler_custom_range.png'")
    plt.show()


def practical_augmentation_example():
    """Show how to use difficulty values for practical augmentation."""
    total_epochs = 50
    
    scheduler = SmoothCurriculumScheduler(
        total_epochs=total_epochs,
        strategy='cosine',
        warmup_epochs=5
    )
    
    print("\n" + "="*70)
    print("PRACTICAL EXAMPLE: Image Masking Augmentation")
    print("="*70)
    print("\nUsing difficulty to control mask ratio (10% to 50%):")
    print("-" * 70)
    print(f"{'Epoch':<10} {'Difficulty':<15} {'Mask Ratio':<15} {'Visual':<30}")
    print("-" * 70)
    
    sample_epochs = [0, 5, 10, 20, 30, 40, 49]
    for epoch in sample_epochs:
        difficulty = scheduler.get_difficulty(epoch)
        
        # Use difficulty to interpolate augmentation parameters
        # Example: Mask ratio from 10% (easy) to 50% (hard)
        min_mask_ratio = 0.1
        max_mask_ratio = 0.5
        mask_ratio = min_mask_ratio + difficulty * (max_mask_ratio - min_mask_ratio)
        
        # Visual bar
        bar_length = int(difficulty * 30)
        visual_bar = '█' * bar_length + '░' * (30 - bar_length)
        
        print(f"{epoch:<10} {difficulty:<15.4f} {mask_ratio:<15.2%} {visual_bar}")
    
    print("-" * 70)
    print("\nOther practical applications:")
    print("  • Text augmentation: synonym_ratio = 0.1 + difficulty * 0.3")
    print("  • Image rotation: max_angle = difficulty * 30  # 0° to 30°")
    print("  • Noise injection: noise_std = 0.01 + difficulty * 0.09  # 0.01 to 0.1")
    print("  • Dropout rate: dropout = 0.1 + difficulty * 0.4  # 0.1 to 0.5")
    print()


def show_schedule_info():
    """Display schedule information for different configurations."""
    print("\n" + "="*70)
    print("SCHEDULER CONFIGURATION EXAMPLES")
    print("="*70)
    
    configs = [
        {
            'name': 'Smooth Linear',
            'scheduler': SmoothCurriculumScheduler(total_epochs=100, strategy='linear')
        },
        {
            'name': 'Smooth Cosine with Warmup',
            'scheduler': SmoothCurriculumScheduler(
                total_epochs=100, strategy='cosine', warmup_epochs=10
            )
        },
        {
            'name': 'Smooth Exponential',
            'scheduler': SmoothCurriculumScheduler(
                total_epochs=100, strategy='exponential', gamma=0.05
            )
        },
        {
            'name': 'Discrete (Original)',
            'scheduler': CurriculumLearningScheduler(total_epochs=100)
        },
        {
            'name': 'Discrete with Smooth Transition',
            'scheduler': CurriculumLearningScheduler(
                total_epochs=100, smooth_transition=True
            )
        }
    ]
    
    for config in configs:
        print(f"\n{config['name']}:")
        print("-" * 70)
        info = config['scheduler'].get_schedule_info()
        for key, value in info.items():
            if key == 'schedule':
                print(f"  {key}:")
                for item in value:
                    print(f"    - {item}")
            elif key == 'samples':
                print(f"  {key}:")
                for epoch, diff in value.items():
                    print(f"    {epoch}: {diff}")
            elif key == 'parameters' and value:
                print(f"  {key}:")
                for param, val in value.items():
                    if val is not None:
                        print(f"    {param}: {val}")
            else:
                print(f"  {key}: {value}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("CURRICULUM LEARNING SCHEDULER EXAMPLES")
    print("="*70)
    
    # Show configuration info
    show_schedule_info()
    
    # Show practical example
    practical_augmentation_example()
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*70 + "\n")
    
    try:
        plot_scheduler_comparison()
        demonstrate_warmup()
        demonstrate_custom_range()
    except ImportError:
        print("⚠ Matplotlib not installed. Skipping plot generation.")
        print("  Install with: pip install matplotlib")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
