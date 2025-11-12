# Text Augmentation Module

This module provides text augmentation strategies for VQA with curriculum learning support.

## Overview

We have **2 text augmentation approaches**:

### 1. `simple.py` - **SimpleTextAugmentation** (Language-Agnostic)

```python
from augmentation import AugmentationFactory, DifficultyLevel

# Basic word-level operations
augmentor = AugmentationFactory.create_text_augmentation(
    augmentation_type='simple',
    difficulty=DifficultyLevel.MEDIUM
)
augmented = augmentor.augment("Có bao nhiêu người trong ảnh?")
```

**Characteristics:**
- ✅ **New OOP architecture** - Inherits from `BaseTextAugmentation`
- ✅ Curriculum learning support (EASY/MEDIUM/HARD)
- ✅ Difficulty-based parameters
- ✅ Integrated with factory pattern
- ✅ Simple, language-agnostic operations:
  - Random word deletion
  - Random word swapping
  - Basic synonym replacement (question words)
- 🌍 **Works for any language**

**Difficulty Levels:**
- **EASY**: 5% deletion, 5% swap, 10% synonym
- **MEDIUM**: 10% deletion, 10% swap, 20% synonym
- **HARD**: 15% deletion, 15% swap, 30% synonym

**When to use:**
- Multi-language support (works for any language)
- Simple baseline text augmentation
- When you want progressive difficulty without language-specific rules

---

### 2. `simple.py` - **RuleBasedTextAugmentation** (Vietnamese-Specific) ⭐

```python
from augmentation import AugmentationFactory, DifficultyLevel

# Rich Vietnamese linguistic rules
augmentor = AugmentationFactory.create_text_augmentation(
    augmentation_type='rule-based',
    difficulty=DifficultyLevel.MEDIUM
)
augmented = augmentor.augment("Có bao nhiêu người trong ảnh?")
```

**Characteristics:**
- ✅ **New OOP architecture** - Inherits from `BaseTextAugmentation`
- ✅ Curriculum learning support
- ✅ **Rich Vietnamese linguistic rules** (from `rule_based.py`):
  - Question word variations (gì, cái gì, thứ gì)
  - Color synonyms (đỏ, đỏ thẫm, son)
  - Verb synonyms (có, có phải, có phải là)
  - Demonstratives (này, đây, nầy)
  - Adjective synonyms (lớn, to, rộng)
  - Vietnamese-specific paraphrasing
- ✅ Difficulty controls **application frequency**:
  - EASY: 20% chance to apply augmentation
  - MEDIUM: 50% chance to apply augmentation
  - HARD: 80% chance to apply augmentation
- 🇻🇳 **Vietnamese-specific**

**When to use:** ⭐ **RECOMMENDED for Vietnamese VQA**
- Training with curriculum learning
- Vietnamese-specific VQA tasks
- Want rich linguistic rules AND curriculum support
- Production training pipelines

---

## Comparison Table

| Feature | `SimpleTextAugmentation` | `RuleBasedTextAugmentation` ⭐ |
|---------|--------------------------|-------------------------------|
| **OOP Architecture** | ✅ Yes | ✅ Yes |
| **Curriculum Learning** | ✅ Yes | ✅ Yes |
| **Factory Pattern** | ✅ Yes | ✅ Yes |
| **Vietnamese Rules** | ⚠️ Basic | ✅ Rich |
| **Difficulty Levels** | ✅ 3 levels | ✅ 3 levels |
| **Language Support** | 🌍 Any language | 🇻🇳 Vietnamese only |
| **Use Case** | Simple/Multi-lang | **Production Vietnamese** |

---

## Usage Examples

### Example 1: Simple Text Augmentation

```python
from augmentation import AugmentationFactory, DifficultyLevel

# Create augmentor
augmentor = AugmentationFactory.create_text_augmentation(
    augmentation_type='simple',
    difficulty=DifficultyLevel.EASY,
    seed=42
)

# Apply augmentation
original = "Có bao nhiêu người trong ảnh?"
augmented = augmentor.augment(original)

print(f"Original:  {original}")
print(f"Augmented: {augmented}")
# Possible output: "Có mấy người trong ảnh?" (synonym replacement)
```

### Example 2: Rule-Based Text Augmentation ⭐

```python
from augmentation import AugmentationFactory, DifficultyLevel

# Create rule-based augmentor (rich Vietnamese rules)
augmentor = AugmentationFactory.create_text_augmentation(
    augmentation_type='rule-based',
    difficulty=DifficultyLevel.MEDIUM,
    seed=42
)

# Apply augmentation
original = "Có bao nhiêu người trong ảnh?"
augmented = augmentor.augment(original)

print(f"Original:  {original}")
print(f"Augmented: {augmented}")
# Possible output: "Có mấy người ở trong hình?" (rich Vietnamese paraphrasing)
```

### Example 3: Curriculum Learning

```python
from augmentation import AugmentationFactory, DifficultyLevel

# Create augmentor with EASY difficulty
augmentor = AugmentationFactory.create_text_augmentation(
    augmentation_type='rule-based',
    difficulty=DifficultyLevel.EASY,
    seed=42
)

# EASY: Only 20% of texts are augmented
for i in range(10):
    text = "Có bao nhiêu người trong ảnh?"
    augmented = augmentor.augment(text)
    if text != augmented:
        print(f"Augmented: {augmented}")
# Expect ~2 augmentations out of 10

# Change to HARD difficulty
augmentor.set_difficulty(DifficultyLevel.HARD)

# HARD: 80% of texts are augmented
for i in range(10):
    text = "Có bao nhiêu người trong ảnh?"
    augmented = augmentor.augment(text)
    if text != augmented:
        print(f"Augmented: {augmented}")
# Expect ~8 augmentations out of 10
```

### Example 4: Integration with Training

```python
from augmentation import AugmentationFactory, DifficultyLevel
from dataset import ViVQADataset

# Create dataset
dataset = ViVQADataset(...)

# Create text augmentor (rule-based for Vietnamese)
text_augmentor = AugmentationFactory.create_text_augmentation(
    augmentation_type='rule-based',
    difficulty=DifficultyLevel.EASY,
    seed=42
)

# Set augmentation
dataset.set_text_augmentation(lambda text: text_augmentor.augment(text))

# During training, questions are automatically augmented
sample = dataset[0]  # Question is augmented on-the-fly
```

---

## Recommendation for ViVQA Experiments

For the **7 ViVQA experiments**, use:

### `rule-based` augmentation type ⭐

```bash
# Experiments with text augmentation
python train.py \
    --dataset_name vivqa \
    --enable_text_augmentation \
    --text_augmentation_type rule-based \  # ← Rich Vietnamese rules
    --epochs 30
```

**Why?**

- ✅ Rich Vietnamese linguistic rules
- ✅ Curriculum learning support
- ✅ Better paraphrasing for Vietnamese questions
- ✅ Best performance on Vietnamese VQA

### Alternative: Use `simple` for comparison

```bash
# Compare simple vs rule-based augmentation
python train.py \
    --dataset_name vivqa \
    --enable_text_augmentation \
    --text_augmentation_type simple \  # ← Language-agnostic baseline
    --epochs 30
```

**Why?**

- ✅ Good baseline for text augmentation
- ✅ Language-agnostic (works for any language)
- ✅ Can compare effectiveness of Vietnamese-specific rules

---

## Summary

**Two Approaches:**

1. **Simple** (`SimpleTextAugmentation`): Language-agnostic, basic word operations
2. **Rule-Based** (`RuleBasedTextAugmentation`): Vietnamese-specific, rich linguistic rules ⭐

**For Vietnamese VQA:**

- Use `rule-based` augmentation type
- Gets rich Vietnamese linguistic rules from `rule_based.py` with OOP architecture
- Perfect for production Vietnamese VQA training

**Implementation Note:**

- `RuleBasedTextAugmentation` uses `VietnameseVQAAugmentation` from `rule_based.py` internally
- Wraps it in OOP architecture with curriculum learning support
- `rule_based.py` is kept as internal implementation detail
