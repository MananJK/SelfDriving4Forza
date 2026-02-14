import os
import time
import torch
import numpy as np
import cv2
from ImageGrab import grab_screen
import direct_input
import getkeys
from model import DrivingModel, DrivingModelLight

SCREEN_SIZE = 224
MINIMAP_SIZE = 128
GAME_REGION = (0, 40, 1024, 768)


class SmoothingFilter:
    """
    Exponential moving average filter for smooth steering outputs.
    Prevents jerky movements while allowing responsive control.
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0

    def update(self, steering, throttle, brake):
        self.steering = self.alpha * steering + (1 - self.alpha) * self.steering
        self.throttle = self.alpha * throttle + (1 - self.alpha) * self.throttle
        self.brake = self.alpha * brake + (1 - self.alpha) * self.brake
        return self.steering, self.throttle, self.brake

    def reset(self):
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0


def load_model(model_path, device="cpu"):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)

    use_minimap = checkpoint.get("use_minimap", False)

    if use_minimap:
        model = DrivingModel()
    else:
        model = DrivingModelLight()

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, use_minimap


def find_latest_model(model_dir="models"):
    """Find the most recent model file."""
    if not os.path.exists(model_dir):
        return None

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not model_files:
        return None

    # Sort by modification time
    model_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True
    )
    return os.path.join(model_dir, model_files[0])


def preprocess_screen(screen, target_size=(224, 224)):
    """Preprocess screen for model input."""
    screen = cv2.resize(screen, target_size)
    screen = screen.astype(np.float32) / 255.0
    screen = np.transpose(screen, (2, 0, 1))  # HWC to CHW
    return torch.tensor(screen, dtype=torch.float32).unsqueeze(0)


def preprocess_minimap(minimap, target_size=(128, 128)):
    """Preprocess minimap for model input."""
    minimap = cv2.resize(minimap, target_size)
    minimap = minimap.astype(np.float32) / 255.0
    minimap = np.transpose(minimap, (2, 0, 1))
    return torch.tensor(minimap, dtype=torch.float32).unsqueeze(0)


def run_self_driving(model_path=None, use_minimap=False, device=None):
    """
    Run the model in self-driving mode.

    Args:
        model_path: Path to model checkpoint (auto-detected if None)
        use_minimap: Whether to use minimap input
        device: torch device (auto-detected if None)
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find model
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            print("No model found! Train a model first.")
            return
        print(f"Using model: {model_path}")

    # Load model
    model, model_uses_minimap = load_model(model_path, device)
    use_minimap = use_minimap and model_uses_minimap
    print(f"Model uses minimap: {model_uses_minimap}")

    # Initialize controller and smoothing
    controller = direct_input.XboxController()
    smoother = SmoothingFilter(alpha=0.4)

    # Minimap settings
    minimap_x = 0
    minimap_y = GAME_REGION[3] - 250
    minimap_size = 250

    print("\nSelf-Driving Mode")
    print("=" * 50)
    print("Controls:")
    print("  - Q: Quit")
    print("  - P: Pause/Resume")
    print("  - Arrow keys: Adjust minimap position")
    print("=" * 50)

    # Countdown
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    print("Running! Press 'Q' to quit, 'P' to pause.")

    paused = False
    last_time = time.time()
    fps_counter = 0
    fps = 0
    start_time = time.time()

    try:
        while True:
            # Check keys
            keys = getkeys.key_check()

            if "Q" in keys:
                print("Quitting...")
                break

            if "P" in keys:
                paused = not paused
                print("Paused" if paused else "Resumed")
                time.sleep(0.3)

            # Adjust minimap position
            if "LEFT" in keys:
                minimap_x = max(0, minimap_x - 10)
            if "RIGHT" in keys:
                minimap_x = min(GAME_REGION[2] - minimap_size, minimap_x + 10)
            if "UP" in keys:
                minimap_y = max(0, minimap_y - 10)
            if "DOWN" in keys:
                minimap_y = min(GAME_REGION[3] - minimap_size, minimap_y + 10)

            if not paused:
                # Grab screen
                screen = grab_screen(region=GAME_REGION)

                # Preprocess screen
                screen_tensor = preprocess_screen(
                    screen, (SCREEN_SIZE, SCREEN_SIZE)
                ).to(device)

                # Get minimap if needed
                if use_minimap:
                    minimap = screen[
                        minimap_y : minimap_y + minimap_size,
                        minimap_x : minimap_x + minimap_size,
                    ]
                    minimap_tensor = preprocess_minimap(
                        minimap, (MINIMAP_SIZE, MINIMAP_SIZE)
                    ).to(device)

                # Run inference
                with torch.no_grad():
                    if use_minimap:
                        steering, throttle, brake = model(screen_tensor, minimap_tensor)
                    else:
                        steering, throttle, brake = model(screen_tensor)

                    steering = steering.item()
                    throttle = throttle.item()
                    brake = brake.item()

                # Apply smoothing
                steering, throttle, brake = smoother.update(steering, throttle, brake)

                # Apply controls
                controller.set_steering(steering)
                controller.set_throttle(throttle)
                controller.set_brake(brake)

                # FPS calculation
                fps_counter += 1
                if time.time() - start_time >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    start_time = time.time()

                # Create display
                display = cv2.resize(screen, (640, 480))

                # Draw controls overlay
                cv2.putText(
                    display,
                    f"FPS: {fps}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display,
                    f"Steering: {steering:.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display,
                    f"Throttle: {throttle:.2f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display,
                    f"Brake: {brake:.2f}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Draw steering indicator
                center_x = display.shape[1] // 2
                center_y = display.shape[0] - 50
                steering_x = int(center_x + steering * 100)
                cv2.circle(display, (center_x, center_y), 30, (100, 100, 100), 2)
                cv2.circle(display, (steering_x, center_y), 10, (0, 255, 255), -1)

                cv2.imshow("Self-Driving", display)
            else:
                # Paused - release all controls
                controller.set_steering(0)
                controller.set_throttle(0)
                controller.set_brake(0)
                smoother.reset()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        # Cleanup
        controller.set_steering(0)
        controller.set_throttle(0)
        controller.set_brake(0)
        cv2.destroyAllWindows()


def evaluate_model(model_path=None, data_path=None, device=None):
    """
    Evaluate model on a dataset and print metrics.
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find model
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            print("No model found!")
            return

    # Load model
    model, use_minimap = load_model(model_path, device)
    print(f"Loaded model from {model_path}")

    # Load data
    if data_path is None:
        import glob

        data_files = glob.glob("training_data_*.npy") + glob.glob(
            "merged_training_data.npy"
        )
        if not data_files:
            print("No training data found!")
            return
        data_path = data_files[0]

    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)

    # Evaluate
    steering_errors = []
    throttle_errors = []
    brake_errors = []

    model.eval()
    with torch.no_grad():
        for sample in data:
            if len(sample) >= 3:
                screen = sample[0]
                minimap = sample[1] if use_minimap else None
                true_steering, true_throttle, true_brake = sample[2]

                # Preprocess
                screen_tensor = preprocess_screen(screen).to(device)

                if use_minimap and minimap is not None:
                    minimap_tensor = preprocess_minimap(minimap).to(device)
                    pred_steering, pred_throttle, pred_brake = model(
                        screen_tensor, minimap_tensor
                    )
                else:
                    pred_steering, pred_throttle, pred_brake = model(screen_tensor)

                steering_errors.append((pred_steering.item() - true_steering) ** 2)
                throttle_errors.append((pred_throttle.item() - true_throttle) ** 2)
                brake_errors.append((pred_brake.item() - true_brake) ** 2)

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Samples evaluated: {len(steering_errors)}")
    print(f"Steering MSE: {np.mean(steering_errors):.4f}")
    print(f"Throttle MSE: {np.mean(throttle_errors):.4f}")
    print(f"Brake MSE: {np.mean(brake_errors):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test driving model")
    parser.add_argument(
        "--model", type=str, default=None, help="Path to model checkpoint"
    )
    parser.add_argument("--minimap", action="store_true", help="Use minimap input")
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate model on dataset"
    )
    parser.add_argument(
        "--data", type=str, default=None, help="Path to evaluation data"
    )

    args = parser.parse_args()

    if args.evaluate:
        evaluate_model(args.model, args.data)
    else:
        run_self_driving(args.model, args.minimap)
