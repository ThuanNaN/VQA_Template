# Quick Run Guide - ViVQA Augmentation Experiments

## 🚀 Run All Experiments

```bash
cd /home/thuannd/Repository/VQA_Template
./experiments/run_vivqa_augmentation_experiments.sh
```

## 📊 Experiments Included

| # | Name | Text Aug | Image Aug | Curriculum | Command |
|---|------|----------|-----------|------------|---------|
| 1 | Baseline | ❌ | ❌ | ❌ | `exp1_baseline` |
| 2 | Text Only | ✅ | ❌ | ❌ | `exp2_text_augment` |
| 3 | Image Only | ❌ | ✅ | ❌ | `exp3_image_augment` |
| 4 | Text + Image | ✅ | ✅ | ❌ | `exp4_text_image_augment` |
| 5 | Text + CL | ✅ | ❌ | ✅ | `exp5_text_augment_cl` |
| 6 | Image + CL | ❌ | ✅ | ✅ | `exp6_image_augment_cl` |
| 7 | Full (⭐ Best) | ✅ | ✅ | ✅ | `exp7_full_augment_cl` |

## ⚙️ Configuration

- **Dataset**: ViVQA
- **Epochs**: 30 per experiment
- **Batch Size**: 64
- **Learning Rate**: 1e-4
- **Curriculum**: 9 easy + 9 medium + 12 hard epochs
- **Output**: `runs/vivqa_augmentation_experiments/`

## 📈 Monitor Progress

### WandB (if enabled)
Visit: https://wandb.ai
Project: `VQA-Augmentation-Experiments`

### Local Logs
```bash
# Watch specific experiment
tail -f runs/vivqa_augmentation_experiments/exp1_baseline/logs/train.log

# Watch all experiments
watch -n 5 'ls -lh runs/vivqa_augmentation_experiments/'
```

## ⏱️ Time Estimate

- Per experiment: ~2-6 hours (depending on GPU)
- Total (7 experiments): ~14-42 hours

**Recommendation**: Run overnight or on weekends

## 🎯 Expected Results

Best to worst (expected):
1. 🥇 Experiment 7: Full augmentation + CL (Best generalization)
2. 🥈 Experiment 6: Image augmentation + CL
3. 🥉 Experiment 4: Text + Image augmentation
4. Experiment 5: Text augmentation + CL
5. Experiment 3: Image augmentation only
6. Experiment 2: Text augmentation only
7. Experiment 1: Baseline (control)

## 🔧 Customize

Edit `experiments/run_vivqa_augmentation_experiments.sh`:

```bash
# Change epochs
EPOCHS=50

# Change batch size (if OOM)
BATCH_SIZE=32

# Disable WandB
ENABLE_WANDB=false

# Change curriculum schedule
EASY_EPOCHS=15
MEDIUM_EPOCHS=15
HARD_EPOCHS=20
```

## 📁 Output Structure

```
runs/vivqa_augmentation_experiments/
├── exp1_baseline/
│   ├── checkpoint-best/       # Best model
│   ├── checkpoint-XXX/         # Periodic checkpoints
│   ├── logs/                   # Training logs
│   └── config.json            # Training config
├── exp2_text_augment/
├── exp3_image_augment/
├── exp4_text_image_augment/
├── exp5_text_augment_cl/
├── exp6_image_augment_cl/
└── exp7_full_augment_cl/
```

## ✅ Verify Setup

Before running, check:

```bash
# Check dataset exists
ls data/vivqa/

# Check script is executable
ls -l experiments/run_vivqa_augmentation_experiments.sh

# Check Python packages
python -c "import torch, transformers; print('✓ Dependencies OK')"
```

## 🆘 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in the script
BATCH_SIZE=32  # or 16
```

### Script Permission Denied
```bash
chmod +x experiments/run_vivqa_augmentation_experiments.sh
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

## 📊 After Completion

1. **Check results directory**:
   ```bash
   ls -lh runs/vivqa_augmentation_experiments/
   ```

2. **Compare on WandB** (if enabled)

3. **Load best model**:
   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained(
       'runs/vivqa_augmentation_experiments/exp7_full_augment_cl/checkpoint-best'
   )
   ```

## 📝 Notes

- All experiments use the same random seed (42) for fair comparison
- Validation is performed after each epoch
- Best model is saved based on validation loss
- Early stopping with patience=5 epochs

---

**Ready to start? Run the script!**

```bash
./experiments/run_vivqa_augmentation_experiments.sh
```
