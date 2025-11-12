"""
Validation tests for Image Augmentation with Curriculum Learning.

This script validates the implementation of MaskedImageAugmentation
and CurriculumLearningScheduler to ensure correct behavior.
"""

import sys
sys.path.append('..')

from augmentation import (
    MaskedImageAugmentation,
    CurriculumLearningScheduler,
    DifficultyLevel,
    create_augmentor_for_epoch
)
from PIL import Image
import numpy as np


def test_difficulty_levels():
    """Test that all difficulty levels can be instantiated."""
    print("Testing difficulty levels...")
    
    for difficulty in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]:
        try:
            augmentor = MaskedImageAugmentation(difficulty=difficulty)
            assert augmentor.difficulty == difficulty
            print(f"  ✓ {difficulty.value.upper()} level created successfully")
        except Exception as e:
            print(f"  ✗ Failed to create {difficulty.value.upper()} level: {e}")
            return False
    
    # Test string-based difficulty
    try:
        augmentor = MaskedImageAugmentation(difficulty="easy")
        assert augmentor.difficulty == DifficultyLevel.EASY
        print(f"  ✓ String-based difficulty creation works")
    except Exception as e:
        print(f"  ✗ Failed string-based difficulty: {e}")
        return False
    
    return True


def test_mask_ratios():
    """Test that mask ratios are set correctly for each difficulty."""
    print("\nTesting mask ratios...")
    
    expected_ratios = {
        DifficultyLevel.EASY: 0.15,
        DifficultyLevel.MEDIUM: 0.50,
        DifficultyLevel.HARD: 0.75
    }
    
    for difficulty, expected_ratio in expected_ratios.items():
        augmentor = MaskedImageAugmentation(difficulty=difficulty)
        actual_ratio = augmentor.mask_ratio
        
        if abs(actual_ratio - expected_ratio) < 0.01:
            print(f"  ✓ {difficulty.value.upper()}: {actual_ratio} (expected {expected_ratio})")
        else:
            print(f"  ✗ {difficulty.value.upper()}: {actual_ratio} (expected {expected_ratio})")
            return False
    
    # Test custom mask ratio
    custom_ratio = 0.3
    augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.EASY, mask_ratio=custom_ratio)
    if abs(augmentor.mask_ratio - custom_ratio) < 0.01:
        print(f"  ✓ Custom mask ratio: {augmentor.mask_ratio}")
    else:
        print(f"  ✗ Custom mask ratio failed")
        return False
    
    return True


def test_augmentation_output():
    """Test that augmentation produces valid outputs."""
    print("\nTesting augmentation output...")
    
    # Create a test image
    test_image = Image.new('RGB', (224, 224), color='blue')
    
    augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.MEDIUM, seed=42)
    
    # Test full augmentation
    try:
        augmented = augmentor.augment(test_image)
        assert isinstance(augmented, Image.Image)
        assert augmented.size == test_image.size
        print(f"  ✓ Full augmentation produces valid output")
    except Exception as e:
        print(f"  ✗ Full augmentation failed: {e}")
        return False
    
    # Test selective augmentations
    selective_tests = [
        ("masking only", {"apply_masking": True, "apply_color_jitter": False, 
                          "apply_blur": False, "apply_brightness": False, "apply_contrast": False}),
        ("color jitter only", {"apply_masking": False, "apply_color_jitter": True,
                               "apply_blur": False, "apply_brightness": False, "apply_contrast": False}),
        ("blur only", {"apply_masking": False, "apply_color_jitter": False,
                       "apply_blur": True, "apply_brightness": False, "apply_contrast": False}),
    ]
    
    for test_name, kwargs in selective_tests:
        try:
            augmented = augmentor.augment(test_image, **kwargs)
            assert isinstance(augmented, Image.Image)
            assert augmented.size == test_image.size
            print(f"  ✓ {test_name} works")
        except Exception as e:
            print(f"  ✗ {test_name} failed: {e}")
            return False
    
    return True


def test_masking_effect():
    """Test that masking actually changes the image."""
    print("\nTesting masking effect...")
    
    # Create a uniform colored image
    test_image = Image.new('RGB', (224, 224), color=(100, 150, 200))
    
    augmentor = MaskedImageAugmentation(difficulty=DifficultyLevel.HARD, seed=42)
    
    # Apply only masking
    masked = augmentor.augment(
        test_image, 
        apply_masking=True,
        apply_color_jitter=False,
        apply_blur=False,
        apply_brightness=False,
        apply_contrast=False,
        apply_flip=False
    )
    
    # Convert to numpy arrays for comparison
    original_array = np.array(test_image)
    masked_array = np.array(masked)
    
    # Check that images are different
    if not np.array_equal(original_array, masked_array):
        # Calculate percentage of changed pixels
        diff = np.any(original_array != masked_array, axis=2)
        changed_pixels = np.sum(diff)
        total_pixels = diff.size
        change_ratio = changed_pixels / total_pixels
        
        print(f"  ✓ Masking changed {change_ratio:.2%} of pixels")
        
        # Verify that roughly the expected ratio was masked (with some tolerance)
        expected_ratio = augmentor.mask_ratio
        if abs(change_ratio - expected_ratio) < 0.15:  # 15% tolerance
            print(f"  ✓ Change ratio matches expected mask ratio (~{expected_ratio:.2%})")
        else:
            print(f"  ⚠ Change ratio {change_ratio:.2%} differs from expected {expected_ratio:.2%}")
        
        return True
    else:
        print(f"  ✗ Masking did not change the image")
        return False


def test_curriculum_scheduler():
    """Test curriculum learning scheduler."""
    print("\nTesting curriculum scheduler...")
    
    # Test default schedule
    total_epochs = 30
    scheduler = CurriculumLearningScheduler(total_epochs=total_epochs)
    
    # Verify total epochs
    if scheduler.total_epochs == total_epochs:
        print(f"  ✓ Total epochs: {total_epochs}")
    else:
        print(f"  ✗ Total epochs mismatch")
        return False
    
    # Test epoch-to-difficulty mapping
    test_cases = [
        (0, DifficultyLevel.EASY),
        (8, DifficultyLevel.EASY),
        (9, DifficultyLevel.MEDIUM),
        (17, DifficultyLevel.MEDIUM),
        (18, DifficultyLevel.HARD),
        (29, DifficultyLevel.HARD),
    ]
    
    for epoch, expected_difficulty in test_cases:
        actual_difficulty = scheduler.get_difficulty_for_epoch(epoch)
        if actual_difficulty == expected_difficulty:
            print(f"  ✓ Epoch {epoch:2d}: {actual_difficulty.value.upper()} (expected {expected_difficulty.value.upper()})")
        else:
            print(f"  ✗ Epoch {epoch:2d}: {actual_difficulty.value.upper()} (expected {expected_difficulty.value.upper()})")
            return False
    
    return True


def test_custom_scheduler():
    """Test custom curriculum scheduler."""
    print("\nTesting custom scheduler...")
    
    total_epochs = 50
    easy_epochs = 20
    medium_epochs = 20
    hard_epochs = 10
    
    try:
        scheduler = CurriculumLearningScheduler(
            total_epochs=total_epochs,
            easy_epochs=easy_epochs,
            medium_epochs=medium_epochs,
            hard_epochs=hard_epochs
        )
        
        if (scheduler.easy_epochs == easy_epochs and
            scheduler.medium_epochs == medium_epochs and
            scheduler.hard_epochs == hard_epochs):
            print(f"  ✓ Custom schedule created: {easy_epochs}/{medium_epochs}/{hard_epochs}")
        else:
            print(f"  ✗ Custom schedule values don't match")
            return False
        
        # Test boundaries
        test_cases = [
            (0, DifficultyLevel.EASY),
            (19, DifficultyLevel.EASY),
            (20, DifficultyLevel.MEDIUM),
            (39, DifficultyLevel.MEDIUM),
            (40, DifficultyLevel.HARD),
            (49, DifficultyLevel.HARD),
        ]
        
        for epoch, expected_difficulty in test_cases:
            actual_difficulty = scheduler.get_difficulty_for_epoch(epoch)
            if actual_difficulty == expected_difficulty:
                print(f"  ✓ Epoch {epoch:2d}: {actual_difficulty.value.upper()}")
            else:
                print(f"  ✗ Epoch {epoch:2d}: {actual_difficulty.value.upper()} (expected {expected_difficulty.value.upper()})")
                return False
        
    except Exception as e:
        print(f"  ✗ Custom scheduler failed: {e}")
        return False
    
    return True


def test_invalid_scheduler():
    """Test that invalid scheduler configurations are rejected."""
    print("\nTesting invalid scheduler configuration...")
    
    try:
        # This should fail: epochs don't sum to total
        scheduler = CurriculumLearningScheduler(
            total_epochs=30,
            easy_epochs=10,
            medium_epochs=10,
            hard_epochs=5  # Only sums to 25
        )
        print(f"  ✗ Invalid scheduler was accepted (should have failed)")
        return False
    except ValueError as e:
        print(f"  ✓ Invalid scheduler correctly rejected")
        return True
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False


def test_create_augmentor_for_epoch():
    """Test the convenience function for creating augmentors."""
    print("\nTesting create_augmentor_for_epoch function...")
    
    scheduler = CurriculumLearningScheduler(total_epochs=30)
    
    test_epochs = [0, 9, 18, 29]
    expected_difficulties = [
        DifficultyLevel.EASY,
        DifficultyLevel.MEDIUM,
        DifficultyLevel.HARD,
        DifficultyLevel.HARD
    ]
    
    for epoch, expected_difficulty in zip(test_epochs, expected_difficulties):
        try:
            augmentor = create_augmentor_for_epoch(epoch, scheduler, seed=42)
            if augmentor.difficulty == expected_difficulty:
                print(f"  ✓ Epoch {epoch:2d}: Created {augmentor.difficulty.value.upper()} augmentor")
            else:
                print(f"  ✗ Epoch {epoch:2d}: Wrong difficulty")
                return False
        except Exception as e:
            print(f"  ✗ Epoch {epoch:2d}: Failed - {e}")
            return False
    
    return True


def test_reproducibility():
    """Test that seeding produces reproducible results."""
    print("\nTesting reproducibility with seed...")
    
    test_image = Image.new('RGB', (224, 224), color='green')
    
    # Create two augmentors with the same seed
    augmentor1 = MaskedImageAugmentation(difficulty=DifficultyLevel.MEDIUM, seed=42)
    augmentor2 = MaskedImageAugmentation(difficulty=DifficultyLevel.MEDIUM, seed=42)
    
    # Apply same augmentation
    aug1 = augmentor1.augment(test_image)
    aug2 = augmentor2.augment(test_image)
    
    # Check if results are identical
    arr1 = np.array(aug1)
    arr2 = np.array(aug2)
    
    if np.array_equal(arr1, arr2):
        print(f"  ✓ Same seed produces identical results")
    else:
        print(f"  ✗ Same seed produces different results")
        return False
    
    # Create augmentor with different seed
    augmentor3 = MaskedImageAugmentation(difficulty=DifficultyLevel.MEDIUM, seed=123)
    aug3 = augmentor3.augment(test_image)
    arr3 = np.array(aug3)
    
    if not np.array_equal(arr1, arr3):
        print(f"  ✓ Different seed produces different results")
    else:
        print(f"  ⚠ Different seed produces same results (might happen rarely)")
    
    return True


def run_all_tests():
    """Run all validation tests."""
    print("="*60)
    print("VALIDATION TESTS FOR IMAGE AUGMENTATION")
    print("="*60)
    
    tests = [
        ("Difficulty Levels", test_difficulty_levels),
        ("Mask Ratios", test_mask_ratios),
        ("Augmentation Output", test_augmentation_output),
        ("Masking Effect", test_masking_effect),
        ("Curriculum Scheduler", test_curriculum_scheduler),
        ("Custom Scheduler", test_custom_scheduler),
        ("Invalid Scheduler", test_invalid_scheduler),
        ("Create Augmentor for Epoch", test_create_augmentor_for_epoch),
        ("Reproducibility", test_reproducibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
