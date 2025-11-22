#!/bin/bash

################################################################################
# VQA Augmentation Experiments - ViVQA Dataset
# 
# This script runs a comprehensive set of experiments to evaluate the impact
# of different augmentation strategies on VQA performance.
#
# Experiments:
# 1. Baseline (no augmentation)
# 2. Text augmentation only
# 3. Image augmentation only
# 4. Text + Image augmentation
# 5. Text augmentation + Curriculum Learning
# 6. Image augmentation + Curriculum Learning
# 7. Text + Image augmentation + Curriculum Learning
################################################################################

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0  # Specify GPU ID(s) to use (e.g., "0", "0,1", "0,1,2,3")

# Configuration
DATASET_NAME="openvivqa"
VIS_MODEL="google/vit-base-patch16-224" # microsoft/beit-base-patch16-224-pt22k-ft22k
TEXT_MODEL="vinai/bartpho-syllable-base" # vinai/bartpho-syllable
TEXT_AGGREGATION="mean" # 'mean', 'sum', 'max', 'first', 'attention', 'transformer', 'gated', 'weighted'
SEED=71
EPOCHS=30
BATCH_SIZE=16
SEQ_LEN=64
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4
GRADIENT_ACCUMULATION=2
WARMUP_STEPS=250
PATIENCE=10
FP16_FLAG="--fp16"
LOGGING_STEPS=50
DATALOADER_WORKERS=4
SAMPLE_OBSERVATION="--enable_sample_observation"

# Augmentation Methods
IMAGE_AUGMENT_METHOD="masked"  # Options: masked, none
TEXT_AUGMENT_METHOD="rule-based"  # Options: rule-based, none

# Curriculum Learning Configuration
CURRICULUM_STRATEGY="linear"  # Options: linear, cosine, exponential, step, polynomial
WARMUP_EPOCHS=0
CURRICULUM_GAMMA=0.1  # For exponential strategy
CURRICULUM_STEP_SIZE=""  # For step strategy (empty means auto: total_epochs // 3)
CURRICULUM_POWER=2.0  # For polynomial strategy

# WandB Configuration (set to true to enable)
ENABLE_WANDB=true
WANDB_PROJECT="VQA-OpenViVQA"

# Output directory
OUTPUT_DIR="runs/openvivqa_augmentation"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

################################################################################
# Experiment Functions
################################################################################

run_experiment() {
    local exp_name=$1
    local description=$2
    shift 2
    local extra_args=("$@")
    
    print_header "Experiment: $exp_name"
    print_info "$description"
    print_info "Started at: $(date)"
    
    # Build command
    local cmd="python train.py \
        --dataset_name $DATASET_NAME \
        --vis_model_name $VIS_MODEL \
        --text_model_name $TEXT_MODEL \
        --seed $SEED \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --seq_len $SEQ_LEN \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --gradient_accumulation $GRADIENT_ACCUMULATION \
        --warmup_steps $WARMUP_STEPS \
        --patience $PATIENCE \
        --logging_steps $LOGGING_STEPS \
        --dataloader_workers $DATALOADER_WORKERS \
        $SAMPLE_OBSERVATION \
        $FP16_FLAG \
        --output_dir $OUTPUT_DIR \
        --run_name $exp_name"
    
    # Add WandB if enabled
    if [ "$ENABLE_WANDB" = true ]; then
        cmd="$cmd --report_to_wandb --wandb_name $WANDB_PROJECT"
    fi
    
    # Add extra arguments
    for arg in "${extra_args[@]}"; do
        cmd="$cmd $arg"
    done
    
    print_info "Command: $cmd"
    echo ""
    
    # Run the experiment
    if eval $cmd; then
        print_info "✓ Experiment '$exp_name' completed successfully!"
        print_info "Finished at: $(date)"
    else
        print_error "✗ Experiment '$exp_name' failed!"
        print_error "Check logs for details"
        return 1
    fi
    
    echo ""
}

################################################################################
# Main Experiments
################################################################################

print_header "VQA Augmentation Experiments - ViVQA Dataset"
print_info "Total experiments: 7"
print_info "Dataset: $DATASET_NAME"
print_info "Epochs per experiment: $EPOCHS"
print_info "Output directory: $OUTPUT_DIR"
print_info "WandB logging: $ENABLE_WANDB"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

################################################################################
# Experiment 1: Baseline (No Augmentation)
################################################################################

run_experiment \
    "exp1_baseline" \
    "Baseline training without any augmentation" || exit 1

################################################################################
# Experiment 2: Text Augmentation Only
################################################################################

run_experiment \
    "exp2_text_augment" \
    "Training with text augmentation only $TEXT_AUGMENT_METHOD" \
    "--enable_text_augmentation" \
    "--text_aggregation $TEXT_AGGREGATION" \
    "--text_augmentation_type $TEXT_AUGMENT_METHOD" || exit 1

################################################################################
# Experiment 3: Image Augmentation Only
################################################################################

run_experiment \
    "exp3_image_augment" \
    "Training with image augmentation $IMAGE_AUGMENT_METHOD only" \
    "--enable_image_augmentation" \
    "--image_augmentation_type $IMAGE_AUGMENT_METHOD" \
    "--patch_size 16" || exit 1

################################################################################
# Experiment 4: Text + Image Augmentation
################################################################################

run_experiment \
    "exp4_text_image_augment" \
    "Training with both text $TEXT_AUGMENT_METHOD and image $IMAGE_AUGMENT_METHOD augmentation" \
    "--enable_text_augmentation" \
    "--text_augmentation_type $TEXT_AUGMENT_METHOD" \
    "--text_aggregation $TEXT_AGGREGATION" \
    "--enable_image_augmentation" \
    "--image_augmentation_type $IMAGE_AUGMENT_METHOD" \
    "--patch_size 16" || exit 1

################################################################################
# Experiment 5: Text Augmentation + Curriculum Learning
################################################################################

run_experiment \
    "exp5_text_augment_cl" \
    "Training with text augmentation $TEXT_AUGMENT_METHOD and curriculum learning ($CURRICULUM_STRATEGY)" \
    "--enable_text_augmentation" \
    "--text_augmentation_type $TEXT_AUGMENT_METHOD" \
    "--text_aggregation $TEXT_AGGREGATION" \
    "--enable_curriculum" \
    "--curriculum_strategy $CURRICULUM_STRATEGY" \
    "--warmup_epochs $WARMUP_EPOCHS" || exit 1

################################################################################
# Experiment 6: Image Augmentation + Curriculum Learning
################################################################################

run_experiment \
    "exp6_image_augment_cl" \
    "Training with image augmentation $IMAGE_AUGMENT_METHOD and curriculum learning ($CURRICULUM_STRATEGY)" \
    "--enable_image_augmentation" \
    "--image_augmentation_type $IMAGE_AUGMENT_METHOD" \
    "--patch_size 16" \
    "--enable_curriculum" \
    "--curriculum_strategy $CURRICULUM_STRATEGY" \
    "--warmup_epochs $WARMUP_EPOCHS" || exit 1

################################################################################
# Experiment 7: Text + Image Augmentation + Curriculum Learning
################################################################################

run_experiment \
    "exp7_full_augment_cl" \
    "Training with text $TEXT_AUGMENT_METHOD + image $IMAGE_AUGMENT_METHOD augmentation and curriculum learning ($CURRICULUM_STRATEGY)" \
    "--enable_text_augmentation" \
    "--text_augmentation_type $TEXT_AUGMENT_METHOD" \
    "--text_aggregation $TEXT_AGGREGATION" \
    "--enable_image_augmentation" \
    "--image_augmentation_type $IMAGE_AUGMENT_METHOD" \
    "--patch_size 16" \
    "--enable_curriculum" \
    "--curriculum_strategy $CURRICULUM_STRATEGY" \
    "--warmup_epochs $WARMUP_EPOCHS" || exit 1

################################################################################
# Summary
################################################################################

print_header "Experiment Suite Completed!"
print_info "All experiments finished"
print_info "Results saved in: $OUTPUT_DIR"

if [ "$ENABLE_WANDB" = true ]; then
    print_info "View results on WandB: https://wandb.ai (Project: $WANDB_PROJECT)"
fi

echo ""
print_info "Completed experiments:"
echo "  ✓ Experiment 1: Baseline (no augmentation)"
echo "  ✓ Experiment 2: Text augmentation"
echo "  ✓ Experiment 3: Image augmentation"
echo "  ✓ Experiment 4: Text + Image augmentation"
echo "  ✓ Experiment 5: Text augmentation + CL"
echo "  ✓ Experiment 6: Image augmentation + CL"
echo "  ✓ Experiment 7: Full augmentation + CL"

echo ""
print_info "To analyze results, check the following locations:"
echo "  - Model checkpoints: $OUTPUT_DIR/exp*/"
echo "  - Training logs: $OUTPUT_DIR/exp*/logs/"
if [ "$ENABLE_WANDB" = true ]; then
    echo "  - WandB dashboard: https://wandb.ai"
fi

echo ""
print_header "Done!"
