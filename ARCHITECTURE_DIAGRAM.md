# OOP Architecture Diagram

## System Overview

```mermaid
flowchart TB
    %% Title
    subgraph VQA["VQA Training System<br/>(OOP Architecture)"]
    end

    %% Modules
    A[Augmentation<br/>Module]
    B[Dataset<br/>Module]
    C[Training<br/>Module]

    %% Connections
    VQA --> A
    VQA --> B
    VQA --> C
```

## Augmentation Module

```mermaid
classDiagram
    %% =======================
    %% Base Classes
    %% =======================
    class BaseAugmentation {
        +__init__(difficulty, seed)
        +set_difficulty(difficulty)
        #_configure_parameters()* abstract
        +augment(data)* abstract
    }

    class BaseImageAugmentation {
        +augment(image: PIL.Image)* abstract
    }

    class BaseTextAugmentation {
        +augment(text: str)* abstract
    }

    BaseImageAugmentation --|> BaseAugmentation
    BaseTextAugmentation --|> BaseAugmentation

    %% =======================
    %% Factory Pattern
    %% =======================
    class AugmentationFactory {
        +create_image_augmentation()
        +create_text_augmentation()
        +register_image_augmentation()
        +register_text_augmentation()
    }

    %% =======================
    %% Visual / Mask
    %% =======================
    class MaskedImageAugmentation {
        +_configure_parameters()
        +augment(image)
        +_random_masking()
    }

    class CurriculumLearningScheduler {
        +get_difficulty_for_epoch(epoch)
        +get_schedule_info()
    }

    MaskedImageAugmentation --|> BaseImageAugmentation

    %% =======================
    %% Package structure (optional grouping)
    %% =======================
    class AugmentationModule {
    }

    AugmentationModule --> BaseAugmentation
    AugmentationModule --> BaseImageAugmentation
    AugmentationModule --> BaseTextAugmentation
    AugmentationModule --> AugmentationFactory
    AugmentationModule --> MaskedImageAugmentation
    AugmentationModule --> CurriculumLearningScheduler
```

## Dataset Module

```mermaid
classDiagram
    %% =======================
    %% Base Dataset
    %% =======================
    class BaseDataset {
        +__init__(ann_path, img_dir, processors, augmentation)
        +__getitem__(idx)
        +set_image_augmentation(augmentor)
        +set_text_augmentation(augmentor)
        %% # Applies augmentation inside __getitem__
    }

    %% =======================
    %% Derived Datasets
    %% =======================
    class ViVQADataset
    class OpenViVQADataset
    class ViTextVQADataset
    class EVJVQADataset
    class ViVQAXDataset
    class ViOCRVQADataset

    %% Inheritance relationships
    ViVQADataset --|> BaseDataset
    OpenViVQADataset --|> BaseDataset
    ViTextVQADataset --|> BaseDataset
    EVJVQADataset --|> BaseDataset
    ViVQAXDataset --|> BaseDataset
    ViOCRVQADataset --|> BaseDataset
```

## Training Module

```mermaid
classDiagram
    %% ==================================
    %% CONFIGURATION MODULE
    %% ==================================
    class ModelConfig
    class DataConfig
    class AugmentationConfig
    class TrainingConfig

    class ExperimentConfig {
        +from_args(args)
    }

    %% ==================================
    %% TRAINER MODULE
    %% ==================================
    class VQATrainer {
        +__init__(curriculum_scheduler, augmentation_factory)
        +_setup_curriculum_learning()
        +log()
    }

    class CurriculumLearningCallback {
        +on_epoch_begin()  %% updates augmentation difficulty
    }

    VQATrainer --|> transformers.Trainer
    CurriculumLearningCallback --|> TrainerCallback

    %% ==================================
    %% PIPELINE MODULE
    %% ==================================
    class VQATrainingPipeline {
        +__init__(config)
        +setup_environment()
        +create_processors()
        +create_datasets()
        +create_model()
        +create_trainer()
        +run()
    }

    class DatasetFactory {
        +create_dataset(dataset_name, ...)
    }

    %% ==================================
    %% RELATIONSHIPS BETWEEN MODULES
    %% ==================================
    VQATrainingPipeline --> VQATrainer : creates
    VQATrainingPipeline --> DatasetFactory : uses
    VQATrainingPipeline --> ExperimentConfig : configures
    VQATrainer --> CurriculumLearningCallback : uses
    VQATrainer --> AugmentationConfig : uses
```

## Data Flow During Training

```mermaid
flowchart TB

%% ========================
%% 1️⃣ Initialization Phase
%% ========================
subgraph P1["1. Initialization Phase"]
    A1["Parse Args"];
    A2["Create Config<br/>(ExperimentCfg)"];
    A3["Create Pipeline"];
    A1 --> A2 --> A3;
end

%% ========================
%% 2️⃣ Setup Phase
%% ========================
subgraph P2["2. Setup Phase"]
    B0["VQATrainingPipeline.run()"];
    B1["Create Processors"];
    B2["Create Datasets"];
    B3["Create Model"];
    B4["Create Curriculum Scheduler"];
    B5["Create Augmentation Factory"];
    B6["Create VQATrainer"];
    B0 --> B1 --> B2 --> B3 --> B4 --> B5 --> B6;
end

A3 --> P2;

%% ========================
%% 3️⃣ Training Loop (Each Epoch)
%% ========================
subgraph P3["3. Training Loop (Each Epoch)"]
    C1["CurriculumLearningCallback.on_epoch_begin()"];
    C2["Get difficulty for current epoch<br/>(EASY → MEDIUM → HARD)"];
    C3["Create augmentor with new difficulty<br/>using augmentation_factory()"];
    C4["Update dataset augmentation<br/>dataset.set_image_augmentation()"];
    C5["Training iterations<br/>(auto-augmentation in __getitem__)"];
    C1 --> C2 --> C3 --> C4 --> C5;
end

P2 --> P3;

%% ========================
%% 4️⃣ Data Loading (Each Batch)
%% ========================
subgraph P4["4. Data Loading (Each Batch)"]
    D1["DataLoader requests batch"];
    D2["BaseDataset.__getitem__(idx)"];
    D3["Load image from disk"];
    D4["Apply augmentation<br/>self.image_augmentation(pil_image)"];
    D5["Apply vision processor"];
    D6["Return batch tensors"];
    D1 --> D2 --> D3 --> D4 --> D5 --> D6;
end

P3 --> P4;

```

## Curriculum Learning Timeline

```plaintext
Epochs:  0────────9────────18───────30
         │        │         │        │
Level:   EASY     MEDIUM    HARD     │
         │        │         │        │
         ▼        ▼         ▼        ▼
Mask:   15%      50%       75%
Blur:   Low      Med       High
Jitter: Min      Mod       Max
```

## Design Patterns Used

```plaintext
┌──────────────────────────────────────────────────┐
│ Factory Pattern                                  │
│ ------------------------------------------------ │
│ AugmentationFactory.create_image_augmentation()  │
│ DatasetFactory.create_dataset()                  │
│                                                  │
│ Benefits:                                        │
│ • Centralized object creation                    │
│ • Easy to register new types                     │
│ • Consistent interface                           │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│ Strategy Pattern                                 │
│ ------------------------------------------------ │
│ Different augmentation strategies:               │
│ • MaskedImageAugmentation                        │
│ • CustomBlurAugmentation                         │
│ • NoAugmentation                                 │
│                                                  │
│ Benefits:                                        │
│ • Interchangeable at runtime                     │
│ • Easy to add new strategies                     │
│ • Follows Open/Closed Principle                  │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│ Template Method Pattern                          │
│ ------------------------------------------------ │
│ BaseAugmentation defines template:               │
│ 1. __init__() calls _configure_parameters()      │
│ 2. Subclasses implement specifics                │
│                                                  │
│ Benefits:                                        │
│ • Consistent initialization                      │
│ • Reusable base logic                            │
│ • Clear extension points                         │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│ Observer Pattern                                 │
│ ------------------------------------------------ │
│ CurriculumLearningCallback observes:             │
│ • Training epoch changes                         │
│ • Updates augmentation accordingly               │
│                                                  │
│ Benefits:                                        │
│ • Loose coupling                                 │
│ • Automatic updates                              │
│ • Easy to extend                                 │
└──────────────────────────────────────────────────┘
```

## Class Relationships

```mermaid
classDiagram
    %% =======================
    %% Base Classes
    %% =======================
    class BaseAugmentation {
        +difficulty
        +seed
        +augment()* 
        +set_difficulty()
    }

    class BaseImageAugmentation {
        +augment(image)* 
    }

    class BaseTextAugmentation {
        +augment(text)*
    }

    class MaskedImageAugmentation {
        +patch_size
        +mask_ratio
        +augment(image)
    }

    class NoAugmentation {
        +augment(data)
    }

    %% =======================
    %% Inheritance
    %% =======================
    BaseImageAugmentation --|> BaseAugmentation
    BaseTextAugmentation --|> BaseAugmentation
    MaskedImageAugmentation --|> BaseImageAugmentation
    NoAugmentation --|> BaseAugmentation

```

## Extension Example

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Class as CustomAugmentation
    participant Factory as AugmentationFactory
    participant Augmentor as Augmentor

    %% Step 1: Create new class
    Dev->>Class: Define _configure_parameters() and augment(image)

    %% Step 2: Register with factory
    Dev->>Factory: register_image_augmentation('custom', CustomAugmentation)

    %% Step 3: Use it
    Dev->>Factory: create_image_augmentation('custom', difficulty='medium')
    Factory->>Augmentor: Return instance of CustomAugmentation
    Dev->>Augmentor: Call augment(image)

```

## Summary

The OOP architecture provides:

✅ **Modularity** - Independent, reusable components
✅ **Extensibility** - Easy to add new features
✅ **Maintainability** - Clear structure and responsibilities
✅ **Type Safety** - Strong typing with dataclasses
✅ **Testability** - Each component can be tested independently
✅ **Scalability** - Ready for complex experiments
