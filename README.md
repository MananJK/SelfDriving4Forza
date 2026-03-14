# SelfDriving4Forza

An end-to-end autonomous driving agent for Forza Horizon 4 using computer vision and deep learning. The AI learns to drive by watching human gameplay and outputting continuous steering, throttle, and brake values.

## Features

- **Continuous Control**: Regression-based model outputs smooth steering (-1 to 1), throttle (0 to 1), and brake (0 to 1)
- **Multi-Input Architecture**: Optionally combines screen and minimap inputs for better context
- **Data Augmentation**: Brightness, contrast, and blur variations for robustness
- **Smooth Steering**: Exponential moving average filter prevents jerky movements
- **Modern Stack**: PyTorch-based training with GPU support

### Advanced Features (New!)

- **Pretrained ResNet18 Backbone**: Transfer learning from ImageNet for better feature extraction
- **Temporal LSTM**: 5-frame history for motion understanding and trajectory prediction
- **Route Detection**: Specialized module for following the blue route line on minimap
- **PID Controller**: Proportional-Integral-Derivative control for precise steering
- **Hybrid Control**: Blends neural network predictions with rule-based route following
- **Trajectory Prediction**: Predicts future steering angles for anticipatory control

## Requirements

- Windows 10/11
- Python 3.8+
- Forza Horizon 4 (1024x768 windowed mode recommended)
- CUDA-capable GPU (optional, for faster training)
- Tesseract OCR (optional, for speedometer reading in demo.py)

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/SelfDriving4Forza.git
cd SelfDriving4Forza
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

```bash
python main.py
```

| Option | Description |
|--------|-------------|
| 1 | Collect training data |
| 2 | Merge multiple data files |
| 3 | Train basic model (screen-only) |
| 4 | Train advanced model (hybrid + temporal) |
| 5 | Test basic model (self-driving mode) |
| 6 | Test advanced model (hybrid control) |
| 7 | Evaluate model on dataset |
| 8 | View data statistics |

### Recommended Workflows

**Basic Workflow:**
1. **Collect Data** (Option 1): Drive naturally in the game. Press `R` to start/pause recording. Aim for 30-60 minutes of varied driving.
2. **Train** (Option 3): Train for 50-100 epochs. The model automatically saves the best checkpoint.
3. **Test** (Option 5): Run the trained model in self-driving mode. Press `P` to pause, `Q` to quit.

**Advanced Workflow (Best Results):**
1. **Collect Data** (Option 1): Same as basic - varied driving with minimap visible
2. **Train Advanced** (Option 4): Train for 100+ epochs with pretrained backbone
3. **Test Hybrid** (Option 6): Run with PID control and route detection
   - `Q` - Quit
   - `P` - Pause/Resume
   - `R` - Reset PID controller
   - `+/-` - Adjust route weight
   - Arrow keys - Adjust minimap position

## Architecture

### Basic Models

**DrivingModel** (screen + minimap):
```
Screen Encoder (224x224):          Minimap Encoder (128x128):
  Conv2d(3→24, stride=2)             Conv2d(3→16, stride=2)
  Conv2d(24→36, stride=2)            Conv2d(16→32, stride=2)
  Conv2d(36→48, stride=2)            Conv2d(32→48, stride=2)
  Conv2d(48→64, stride=2)
  Conv2d(64→64)
                ↓
        Concatenated Features
                ↓
         FC(512) → FC(128) → FC(32)
                ↓
     ┌───────────┼───────────┐
     ↓           ↓           ↓
 Steering     Throttle     Brake
   (tanh)     (sigmoid)   (sigmoid)
```

**DrivingModelLight** (screen only): Smaller version for faster CPU inference.

### Advanced Model (Hybrid Architecture)

**AdvancedDrivingModel** combines multiple advanced techniques:

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT STREAMS                              │
├─────────────────────┬───────────────────────────────────────────┤
│   Screen (224x224)  │         Minimap (128x128)                │
│         ↓           │                 ↓                         │
│  ResNet18 Backbone  │       RouteDetector (CNN + Attention)     │
│    (pretrained)     │         ↓                                 │
│         ↓           │    Route Direction + Route Features       │
│   Screen Features   │                                           │
└─────────┬───────────┴──────────────────┬────────────────────────┘
          │                              │
          └──────────────┬───────────────┘
                         ↓
                  Temporal LSTM
               (5-frame history)
                         ↓
              ┌──────────┴──────────┐
              ↓                     ↓
        Trajectory Predictor    Control Heads
         (future steering)    Steering/Throttle/Brake
```

**Key Components:**

1. **PretrainedScreenEncoder**: ResNet18 with frozen early layers, outputs 256-dim features
2. **RouteDetector**: Attention-based CNN for blue route line detection
3. **TemporalLSTM**: 2-layer LSTM for temporal reasoning (256 hidden)
4. **TrajectoryPredictor**: Predicts 10 future steering angles
5. **HybridController**: Blends NN steering with route detection via PID

**Model Sizes:**
- Basic model: ~500K parameters
- Advanced model: ~25M parameters (24.5M trainable)

## Project Structure

```
SelfDriving4Forza/
├── main.py              # CLI menu
├── model.py             # Basic PyTorch model architectures
├── model_advanced.py    # Advanced model (ResNet18 + LSTM + Route Detector)
├── train_pytorch.py     # Basic training pipeline
├── train_advanced.py    # Advanced training (frame stacking + route geometry)
├── test_pytorch.py      # Basic self-driving inference
├── test_advanced.py     # Hybrid self-driving (PID + NN + Route Detection)
├── collect_data.py      # Data collection (continuous labels)
├── demo.py              # Waypoint detection demo
├── draw_lanes.py        # Blue line detection utilities
├── direct_input.py      # Keyboard input simulation
├── ImageGrab.py         # Screen capture
├── getkeys.py           # Key state detection
├── models/              # Saved model checkpoints
│   ├── best_model_*.pt      # Best basic models
│   ├── best_advanced_*.pt    # Best advanced models
│   └── final_*.pt           # Final models
└── training_data_*.npy  # Collected training data
```

## Tips for Better Performance

### Basic Training
1. **Varied Data**: Record on different tracks, weather conditions, and times of day
2. **Balanced Driving**: Include plenty of turns (left and right), not just straight roads
3. **Consistent Minimap Position**: Don't adjust the minimap capture region mid-session
4. **More Data**: 30+ minutes of driving typically works better than 5 minutes
5. **Monitor Loss**: If validation loss plateaus, try reducing learning rate

### Advanced Training
1. **Use Pretrained Backbone**: ResNet18 significantly improves feature extraction
2. **Frame History**: 5+ frames help the model understand motion direction
3. **Train Longer**: Advanced model benefits from 100+ epochs
4. **Learning Rate**: Use the default 1e-4 with cosine annealing
5. **Batch Size**: 16 works well; smaller batches need more gradient accumulation

### Hybrid Control Tuning
1. **Route Weight**: Increase (0.7-0.8) for precise route following
2. **NN Weight**: Increase (0.5-0.6) for complex maneuvers and obstacle avoidance
3. **PID Gains**: Adjust Kp for responsiveness, Kd for stability
4. **Reset PID**: Press 'R' in-game to reset the controller if behavior becomes erratic

## Troubleshooting

- **Model drives straight only**: Data is likely imbalanced. Collect more turning examples.
- **Jerky steering**: Increase smoothing filter alpha in `test_pytorch.py` (line 22)
- **Low FPS**: Use `DrivingModelLight` or reduce input resolution
- **CUDA out of memory**: Reduce batch size or use CPU
- **Import errors**: Ensure all requirements are installed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
