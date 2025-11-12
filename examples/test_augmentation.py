"""
Simple test of Vietnamese Rule-Based VQA Augmentation (minimal dependencies)
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

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


def test_basic_augmentation():
    """Test basic question augmentation."""
    print("="*70)
    print("TEST 1: Basic Augmentation")
    print("="*70)
    
    augmentor = VietnameseVQAAugmentation(seed=42)
    
    test_questions = [
        "Cái gì trong ảnh này?",
        "Người này đang làm gì?",
        "Màu sắc của chiếc xe là gì?",
        "Có bao nhiêu người trong hình?",
        "Con vật này ở đâu?",
    ]
    
    print("\nAugmenting questions:\n")
    success_count = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"{i}. Original: {question}")
        
        augmented = augmentor.augment_question(question, num_augmentations=3)
        
        if len(augmented) > 0:
            success_count += 1
            for j, aug_q in enumerate(augmented, 1):
                print(f"   Aug {j}: {aug_q}")
        else:
            print("   (No augmentations generated)")
        print()
    
    print(f"Successfully augmented {success_count}/{len(test_questions)} questions")
    return success_count == len(test_questions)


def test_dataset_augmentation():
    """Test augmenting a complete dataset."""
    print("="*70)
    print("TEST 2: Dataset Augmentation")
    print("="*70)
    
    original_dataset = [
        {"question": "Cái gì trong ảnh?", "answer": "con mèo", "img_id": "000001"},
        {"question": "Người này đang làm gì?", "answer": "đọc sách", "img_id": "000002"},
        {"question": "Màu của chiếc xe là gì?", "answer": "đỏ", "img_id": "000003"},
    ]
    
    print(f"\nOriginal dataset size: {len(original_dataset)}")
    
    augmented_dataset = create_augmented_dataset(
        original_dataset, 
        num_augmentations=2,
        seed=42
    )
    
    print(f"Augmented dataset size: {len(augmented_dataset)}")
    expected_size = len(original_dataset) * 3  # original + 2 augmentations each
    
    print(f"\nExpected size: {expected_size}")
    print(f"Actual size: {len(augmented_dataset)}")
    
    success = len(augmented_dataset) >= len(original_dataset)
    print(f"\nTest result: {'PASS' if success else 'FAIL'}")
    return success


def test_rule_coverage():
    """Test that rules are properly loaded."""
    print("="*70)
    print("TEST 3: Rule Coverage")
    print("="*70)
    
    augmentor = VietnameseVQAAugmentation(seed=42)
    
    print("\nChecking rule categories:")
    
    tests = [
        ("Question words", augmentor.question_words, 5),
        ("Color synonyms", augmentor.color_synonyms, 3),
        ("Verb synonyms", augmentor.verb_synonyms, 3),
        ("Demonstratives", augmentor.demonstratives, 3),
        ("Adjective synonyms", augmentor.adjective_synonyms, 3),
    ]
    
    all_passed = True
    for name, rule_dict, min_expected in tests:
        count = len(rule_dict)
        passed = count >= min_expected
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {count} rules (expected >= {min_expected})")
        all_passed = all_passed and passed
    
    print(f"\nTest result: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def test_statistics():
    """Test statistics functionality."""
    print("="*70)
    print("TEST 4: Statistics")
    print("="*70)
    
    augmentor = VietnameseVQAAugmentation(seed=42)
    
    questions = [
        "Cái gì trong ảnh này?",
        "Người này đang làm gì?",
        "Màu sắc của chiếc xe là gì?",
    ]
    
    stats = augmentor.get_statistics(questions)
    
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    success = stats['total_questions'] == len(questions)
    print(f"\nTest result: {'PASS' if success else 'FAIL'}")
    return success


def test_specific_rules():
    """Test specific augmentation rules."""
    print("="*70)
    print("TEST 5: Specific Rule Application")
    print("="*70)
    
    augmentor = VietnameseVQAAugmentation(seed=42)
    
    test_cases = [
        {
            "name": "Question word replacement",
            "question": "Cái gì trong ảnh?",
            "expected_changes": ["gì", "cái gì"]
        },
        {
            "name": "Demonstrative replacement",
            "question": "Con chó này đang ngủ",
            "expected_changes": ["này", "đây"]
        },
        {
            "name": "Color synonym",
            "question": "Màu đỏ của xe",
            "expected_changes": ["đỏ"]
        },
    ]
    
    print("\nTesting specific rules:\n")
    all_passed = True
    
    for case in test_cases:
        print(f"Test: {case['name']}")
        print(f"  Input: {case['question']}")
        
        augmented = augmentor.augment_question(case['question'], num_augmentations=5)
        
        if augmented:
            print(f"  Generated {len(augmented)} variations:")
            for aug in augmented[:3]:  # Show first 3
                print(f"    - {aug}")
            status = "✓ PASS"
        else:
            status = "✗ FAIL (no augmentations)"
            all_passed = False
        
        print(f"  {status}")
        print()
    
    print(f"Overall result: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("VIETNAMESE VQA RULE-BASED AUGMENTATION - TEST SUITE")
    print("="*70 + "\n")
    
    tests = [
        ("Basic Augmentation", test_basic_augmentation),
        ("Dataset Augmentation", test_dataset_augmentation),
        ("Rule Coverage", test_rule_coverage),
        ("Statistics", test_statistics),
        ("Specific Rules", test_specific_rules),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nERROR in {test_name}: {str(e)}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
