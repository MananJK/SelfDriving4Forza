import os
import numpy as np
import cv2
import time
import pickle
from getkeys import key_check
from ImageGrab import grab_screen

WIDTH = 224
HEIGHT = 224
MINIMAP_SIZE = 128
GAME_REGION = (0, 40, 1024, 768)


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def countdown(seconds=3):
    for i in range(seconds, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)


def keys_to_continuous(keys):
    """
    Convert key inputs to continuous steering, throttle, brake values.

    Returns:
        tuple: (steering, throttle, brake)
        - steering: -1.0 (full left) to 1.0 (full right)
        - throttle: 0.0 to 1.0
        - brake: 0.0 to 1.0
    """
    steering = 0.0
    throttle = 0.0
    brake = 0.0

    has_w = "W" in keys
    has_s = "S" in keys
    has_a = "A" in keys
    has_d = "D" in keys

    # Steering
    if has_a and not has_d:
        steering = -1.0
    elif has_d and not has_a:
        steering = 1.0

    # Throttle and brake
    if has_w:
        throttle = 1.0
    if has_s:
        brake = 1.0

    return steering, throttle, brake


def collect_training_data():
    """
    Collect training data by recording gameplay and inputs.
    Saves screen, minimap, and continuous control values.
    """
    print("Data Collection Mode")
    print("=" * 50)
    print("Drive around the game naturally.")
    print("The system will record your inputs and screen data for training.")
    print("Controls:")
    print("  - R: Toggle recording (start/pause)")
    print("  - Q: Quit data collection")
    print("  - Arrow keys: Adjust minimap position")
    print("=" * 50)

    countdown(5)

    training_data = []
    recording = False

    # Minimap settings (bottom-left of screen)
    minimap_x = 0
    minimap_y = GAME_REGION[3] - 250
    minimap_size = 250

    print("Adjust minimap with arrow keys. Press 'R' to toggle recording, 'Q' to quit.")

    prev_keys = set()
    last_time = time.time()

    try:
        while True:
            # Grab full screen
            screen = grab_screen(region=GAME_REGION)

            # Extract minimap region
            minimap = screen[
                minimap_y : minimap_y + minimap_size,
                minimap_x : minimap_x + minimap_size,
            ]

            # Handle key presses
            keys = key_check()

            # Exit on Q
            if "Q" in keys:
                break

            # Move minimap box with arrow keys
            if "LEFT" in keys:
                minimap_x = max(0, minimap_x - 10)
            if "RIGHT" in keys:
                minimap_x = min(GAME_REGION[2] - minimap_size, minimap_x + 10)
            if "UP" in keys:
                minimap_y = max(0, minimap_y - 10)
            if "DOWN" in keys:
                minimap_y = min(GAME_REGION[3] - minimap_size, minimap_y + 10)

            # Toggle recording on 'R' key down (edge detection)
            if "R" in keys and "R" not in prev_keys:
                recording = not recording
                print(f"Recording {'started' if recording else 'paused'}")

            # Create display image
            display_img = minimap.copy()

            # Draw status
            status = "REC" if recording else "PAUSED"
            color = (0, 255, 0) if recording else (0, 0, 255)
            cv2.putText(
                display_img,
                f"{status} | Samples: {len(training_data)}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

            # Show minimap region on screen
            cv2.imshow("Minimap Capture", display_img)

            if recording:
                # Resize images
                screen_resized = cv2.resize(screen, (WIDTH, HEIGHT))
                minimap_resized = cv2.resize(minimap, (MINIMAP_SIZE, MINIMAP_SIZE))

                # Get continuous control values
                steering, throttle, brake = keys_to_continuous(keys)

                # Save: [screen, minimap, [steering, throttle, brake]]
                training_data.append(
                    [screen_resized, minimap_resized, [steering, throttle, brake]]
                )

            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

            prev_keys = set(keys)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cv2.destroyAllWindows()

    # Save data
    if training_data:
        print(f"Saving {len(training_data)} samples...")

        timestamp = int(time.time())
        filename = f"training_data_{timestamp}.npy"

        try:
            np.save(filename, np.array(training_data, dtype=object))
            print(f"Saved to {filename}")
        except MemoryError:
            # Fallback to pickle for large datasets
            pkl_filename = f"training_data_{timestamp}.pkl"
            with open(pkl_filename, "wb") as f:
                pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved to {pkl_filename} (pickle format)")

        # Print data statistics
        steerings = [d[2][0] for d in training_data]
        throttles = [d[2][1] for d in training_data]
        brakes = [d[2][2] for d in training_data]

        print("\nData Statistics:")
        print(f"  Steering: mean={np.mean(steerings):.3f}, std={np.std(steerings):.3f}")
        print(
            f"  Throttle: mean={np.mean(throttles):.3f}, active={sum(1 for t in throttles if t > 0.5)}"
        )
        print(
            f"  Brake:    mean={np.mean(brakes):.3f}, active={sum(1 for b in brakes if b > 0.5)}"
        )

        # Count steering distribution
        left_count = sum(1 for s in steerings if s < -0.5)
        straight_count = sum(1 for s in steerings if -0.5 <= s <= 0.5)
        right_count = sum(1 for s in steerings if s > 0.5)

        print(f"\nSteering Distribution:")
        print(f"  Left:     {left_count} ({100 * left_count / len(steerings):.1f}%)")
        print(
            f"  Straight: {straight_count} ({100 * straight_count / len(steerings):.1f}%)"
        )
        print(f"  Right:    {right_count} ({100 * right_count / len(steerings):.1f}%)")
    else:
        print("No data collected")


def merge_data_files(
    file_pattern="training_data_*.npy", output="merged_training_data.npy"
):
    """Merge multiple training data files into one."""
    import glob

    files = glob.glob(file_pattern)
    if not files:
        print(f"No files matching {file_pattern}")
        return

    print(f"Found {len(files)} files to merge")

    all_data = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        all_data.extend(data)
        print(f"  {f}: {len(data)} samples")

    np.save(output, np.array(all_data, dtype=object))
    print(f"Merged {len(all_data)} samples into {output}")


if __name__ == "__main__":
    collect_training_data()
