# Legacy TensorFlow Implementation

This folder contains the original implementation using TensorFlow/tflearn.

## What's Here

| File | Purpose |
|------|---------|
| `nervenet.py` | Original tflearn CNN model (67M+ params, 9-class output) |
| `train_model.py` | Original training script with undersampling |
| `test_model.py` | Original inference with burst steering |
| `balance_data.py` | Data balancing via undersampling (throws away data) |
| `tf_fix.py` | TensorFlow GPU memory configuration |

## Why Replaced

1. **tflearn unmaintained** - Last update 2019, compatibility issues
2. **Overparameterized** - 67M+ params caused overfitting
3. **Discrete outputs** - 9 classes too coarse for smooth driving
4. **Data loss** - balance_data.py threw away most collected data
5. **No temporal context** - Single frame decisions

## Current Implementation

See `model.py`, `model_advanced.py`, and related PyTorch files in the parent directory.

**Improvements:**
- ~500K params (basic) / ~25M params (advanced with ResNet18)
- Continuous steering output (-1 to 1)
- Temporal modeling with LSTM
- PID control for smooth steering
- Route detection integration