```markdown
# filepath: README.md

# Forza Horizon 4 Waypoint‑Based Self‑Driving System

## Overview
This project implements an end‑to‑end self‑driving pipeline for Forza Horizon 4 using only on‑screen cues:
- **Minimap Blue‑Line Detection** for waypoint extraction  
- **Full‑screen & Minimap Data Collection** with human keypresses (W/A/S/D)  
- **Data Balancing** to avoid class imbalance  
- **Convolutional Neural Network (Nervenet)** to map visuals → driving commands  
- **Burst‑based Steering Controller** to eliminate over‑steering oscillation  
- **Evaluation Metrics** (accuracy, precision, recall, F1)

## Features
- Interactive **Minimap Alignment** tool with live overlay  
- Pause/Resume recording (`R`) and Quit (`Q`) controls  
- Separate datasets:
  - `training_data_screen.npy` (full‑screen + controls)  
  - `training_data_minimap.npy` (minimap crop + controls)  
- Balanced splits, model training, test drive, and evaluation via main menu  
- Real‑time self‑driving demo with an XInput‑compatible Xbox controller  

## Prerequisites
- Windows 10/11 with a supported game resolution (e.g. 1024×768)  
- Python 3.8+  
- Visual Studio Code (optional)  
- Packages:
  ```bash
  pip install numpy opencv-python tflearn tensorflow pillow pytesseract
  ```
- Tesseract OCR (for speedometer reading):
  https://github.com/UB-Mannheim/tesseract/wiki

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your‑org/WaypointBasedSelf.git
   cd WaypointBasedSelf
   ```
2. (Optional) Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

## File Structure
```
WaypointBasedSelf/
├── demo.py              # Combined detection & steering demo
├── main.py              # CLI menu: collect, balance, train, test, evaluate
├── draw_lanes.py        # Minimap processing & waypoint detection
├── balance_data.py      # Under/oversampling for balanced classes
├── train_model.py       # CNN training (uses Nervenet)
├── test_model.py        # Self‑driving inference loop
├── nervenet.py          # tflearn CNN architecture
├── direct_input.py      # XboxController wrapper (XInput)
├── ImageGrab.py         # grab_screen utility
├── getkeys.py           # Keyboard polling (W/A/S/D, R, Q)
├── tf_fix.py            # Compatibility patches (tflearn, PIL)
├── model/               # Saved model checkpoints
└── logs/                # TensorBoard logs
```

## Usage

Run the main menu:
```bash
python main.py
```

Select from:
1. **Collect Training Data**  
   - Align minimap (arrow keys), press `R` to start/pause, `Q` to quit.  
2. **Balance Training Data**  
   - Creates `balanced_training_data_screen.npy`.  
3. **Train Model**  
   - Trains for 50 epochs by default → saves to `model/<timestamp>/`.  
4. **Test Model (Self‑Driving Mode)**  
   - Live inference with burst steering (0.5 s per turn).  
5. **Waypoint Detection Only**  
   - Visualize minimap blue‑line waypoints.  
6. **Evaluate Model**  
   - Prints accuracy, precision, recall, F1 on held‑out split.  
7. **Quit**

## Data Collection
- **Full‑screen frames** (160×120) + **Minimap crop** (160×120)  
- One‑hot outputs for:  
  `[W, S, A, D, WA, WD, SA, SD, NO_INPUT]`  
- Stored in:
  - `training_data_screen.npy`  
  - `training_data_minimap.npy`

## Training & Evaluation
1. Balance:  
   ```bash
   python main.py → option 2
   ```
2. Train:  
   ```bash
   python main.py → option 3
   ```
3. Evaluate:  
   ```bash
   python main.py → option 6
   ```

## Self‑Driving Demo
```bash
python demo.py
```
- Adjust minimap, toggle auto‑steering (`R`), quit (`Q`).

## Tips & Troubleshooting
- Ensure your Forza window matches `GAME_REGION` in `main.py`/`demo.py`.  
- If “No saved models found” appears, verify `model/<timestamp>/` contains the `.meta`, `.index`, `.data-*` files.  
- Use `pip install pillow --upgrade` if you encounter PIL resampling errors.

## License & Acknowledgments
This project uses:
- [TensorFlow](https://www.tensorflow.org/)  
- [tflearn](https://github.com/tflearn/tflearn)  
- [OpenCV](https://opencv.org/)  
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)  

© 2025 MananJK / Nerve  
```
