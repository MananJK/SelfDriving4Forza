# SelfDriving4Forza

An end-to-end autonomous driving agent for Forza Horizon 4 using computer vision and deep learning. The AI learns to drive by watching human gameplay and detecting the in-game waypoint route.

## Features

- **Waypoint Detection**: Extracts the blue route line from the minimap using HSV color filtering and contour detection
- **End-to-End CNN**: AlexNet-style neural network maps screen pixels directly to driving commands
- **Burst Steering**: Prevents oscillation by holding steering inputs for fixed durations
- **Data Collection**: Record your own driving data with synchronized screen/key captures
- **Class Balancing**: Undersampling pipeline prevents model bias toward common actions

## Requirements

- Windows 10/11
- Python 3.8+
- Forza Horizon 4 (1024x768 windowed mode recommended)
- Tesseract OCR (optional, for speedometer reading)

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/SelfDriving4Forza.git
cd SelfDriving4Forza
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the main CLI:
```bash
python main.py
```

| Option | Description |
|--------|-------------|
| 1 | Collect training data (WASD + screen capture) |
| 2 | Balance collected data across action classes |
| 3 | Train the Nervenet CNN model |
| 4 | Test model in self-driving mode |
| 5 | Waypoint detection visualization only |
| 6 | Evaluate model metrics |

### Standalone Demo
```bash
python demo.py
```

## Architecture

**Nervenet** (AlexNet-style CNN):
- Input: 160×120×3 RGB image
- 5 Conv layers with max pooling
- 2 Fully connected layers (4096 units each)
- Output: 9-class softmax (W, S, A, D, WA, WD, SA, SD, none)

## Project Structure

```
SelfDriving4Forza/
├── main.py           # CLI menu
├── demo.py           # Standalone waypoint demo
├── train_model.py    # Model training
├── test_model.py     # Self-driving inference
├── nervenet.py       # CNN architecture
├── draw_lanes.py     # Blue line detection
├── balance_data.py   # Dataset balancing
├── direct_input.py   # Keyboard input simulation
├── ImageGrab.py      # Screen capture
├── getkeys.py        # Key state detection
└── tf_fix.py         # TensorFlow GPU config
```

## Troubleshooting

- **Game window not found**: Adjust `GAME_REGION` in `main.py` (line 20)
- **No models found**: Train a model first (option 3)
- **Pillow resampling errors**: `pip install --upgrade pillow`
- **GPU memory errors**: Already handled by `tf_fix.py`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
