# SelfDriving4Forza

An end-to-end autonomous driving agent for Forza Horizon 4 using computer vision and deep learning. The AI learns to drive by watching human gameplay and outputting continuous steering, throttle, and brake values.

## Features

- **Continuous Control**: Regression-based model outputs smooth steering (-1 to 1), throttle (0 to 1), and brake (0 to 1)
- **Multi-Input Architecture**: Optionally combines screen and minimap inputs for better context
- **Data Augmentation**: Brightness, contrast, and blur variations for robustness
- **Smooth Steering**: Exponential moving average filter prevents jerky movements
- **Modern Stack**: PyTorch-based training with GPU support

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
| 3 | Train model |
| 4 | Self-driving mode |
| 5 | Evaluate model on dataset |
| 6 | View data statistics |

### Recommended Workflow

1. **Collect Data** (Option 1): Drive naturally in the game. Press `R` to start/pause recording. Aim for 30-60 minutes of varied driving.

2. **Train** (Option 3): Train for 50-100 epochs. The model automatically saves the best checkpoint.

3. **Test** (Option 4): Run the trained model in self-driving mode. Press `P` to pause, `Q` to quit.

## Architecture

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

## Project Structure

```
SelfDriving4Forza/
├── main.py            # CLI menu
├── model.py           # PyTorch model architectures
├── train_pytorch.py   # Training pipeline with augmentation
├── test_pytorch.py    # Self-driving inference
├── collect_data.py    # Data collection (continuous labels)
├── demo.py            # Waypoint detection demo
├── draw_lanes.py      # Blue line detection utilities
├── direct_input.py    # Keyboard input simulation
├── ImageGrab.py       # Screen capture
├── getkeys.py         # Key state detection
├── models/            # Saved model checkpoints
└── training_data_*.npy  # Collected training data
```

## Tips for Better Performance

1. **Varied Data**: Record on different tracks, weather conditions, and times of day
2. **Balanced Driving**: Include plenty of turns (left and right), not just straight roads
3. **Consistent Minimap Position**: Don't adjust the minimap capture region mid-session
4. **More Data**: 30+ minutes of driving typically works better than 5 minutes
5. **Monitor Loss**: If validation loss plateaus, try reducing learning rate

## Troubleshooting

- **Model drives straight only**: Data is likely imbalanced. Collect more turning examples.
- **Jerky steering**: Increase smoothing filter alpha in `test_pytorch.py` (line 22)
- **Low FPS**: Use `DrivingModelLight` or reduce input resolution
- **CUDA out of memory**: Reduce batch size or use CPU
- **Import errors**: Ensure all requirements are installed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
