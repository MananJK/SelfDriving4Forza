import os
import numpy as np
import cv2
import time
import random
from getkeys import key_check
from ImageGrab import grab_screen
import draw_lanes
import tensorflow as tf
import tf_fix
import direct_input
import sys

# Define constants
WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 50
MODEL_NAME = 'forza_waypoint_driver'
GAME_REGION = (0, 40, 1024, 768)  # Updated game region for 1024x768 resolution

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def countdown(seconds=3):
    """Countdown before starting a function"""
    for i in range(seconds, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

def collect_training_data():
    """Collect training data by recording gameplay and inputs"""
    from getkeys import key_check
    
    print("Data Collection Mode")
    print("=" * 50)
    print("Drive around the game and follow the waypoints naturally.")
    print("The system will record your inputs and screen data for training.")
    print("Controls:")
    print("  - R: Toggle recording (start/pause)")
    print("  - Q: Quit data collection")
    print("=" * 50)
    
    # Brief countdown before starting alignment and recording
    countdown(5)
    # Prepare separate datasets for full-screen and minimap recordings
    screen_data = []
    minimap_data = []
    recording = False
    # Initialize minimap alignment region
    minimap_x, minimap_y = 0, GAME_REGION[3] - 250  # start at bottom-left
    minimap_size = 250
    print("Adjust minimap with arrow keys. Press 'R' to toggle recording, 'Q' to quit.")
    last_time = time.time()
    prev_keys = set()

    while True:
        # Grab full screen
        screen = grab_screen(region=GAME_REGION)
        # Crop minimap
        minimap = screen[minimap_y:minimap_y+minimap_size, minimap_x:minimap_x+minimap_size]
        # Detect blue line for alignment
        line_pts, center_line, minimap_debug = draw_lanes.detect_blue_line(minimap, debug=True)
        if minimap_debug is None:
            minimap_debug = minimap.copy()

        # Handle key presses for alignment and recording control
        keys = key_check()
        # Exit on Q
        if 'Q' in keys:
            cv2.destroyWindow('Minimap Setup')
            break
        # Move minimap box
        if 'LEFT' in keys:
            minimap_x = max(0, minimap_x - 10)
        if 'RIGHT' in keys:
            minimap_x = min(GAME_REGION[2] - minimap_size, minimap_x + 10)
        if 'UP' in keys:
            minimap_y = max(0, minimap_y - 10)
        if 'DOWN' in keys:
            minimap_y = min(GAME_REGION[3] - minimap_size, minimap_y + 10)
        # Toggle recording on 'R' key down
        if 'R' in keys and 'R' not in prev_keys:
            recording = not recording
            print("Recording started") if recording else print("Recording paused")

        # Always update overlay with status and sample count (screen data count)
        status = "REC" if recording else "PAUSED"
        cv2.putText(minimap_debug, f"Samples: {len(screen_data)} | {status}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0) if recording else (0,0,255), 1)
        cv2.imshow('Minimap Setup', minimap_debug)
        if recording:
            # Record frame when unpaused into separate datasets
            screen_resized = cv2.resize(screen, (WIDTH, HEIGHT))
            minimap_resized = cv2.resize(minimap, (WIDTH, HEIGHT))
            output = keys_to_output(keys)
            screen_data.append([screen_resized, output])
            minimap_data.append([minimap_resized, output])

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # Update previous keys for edge detection
        prev_keys = set(keys)

    # Save data
    if screen_data:
        import pickle
        # Save full-screen data
        try:
            np.save('training_data_screen.npy', np.array(screen_data, dtype=object))
            print(f"Saved {len(screen_data)} full-screen samples to training_data_screen.npy")
        except MemoryError:
            fallback_scr = 'training_data_screen.pkl'
            print('MemoryError saving full-screen numpy file; falling back to pickle ->', fallback_scr)
            with open(fallback_scr, 'wb') as f:
                pickle.dump(screen_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {len(screen_data)} full-screen samples to {fallback_scr}")
        # Save minimap data
        try:
            np.save('training_data_minimap.npy', np.array(minimap_data, dtype=object))
            print(f"Saved {len(minimap_data)} minimap samples to training_data_minimap.npy")
        except MemoryError:
            fallback_min = 'training_data_minimap.pkl'
            print('MemoryError saving minimap numpy file; falling back to pickle ->', fallback_min)
            with open(fallback_min, 'wb') as f:
                pickle.dump(minimap_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {len(minimap_data)} minimap samples to {fallback_min}")
    cv2.destroyAllWindows()

def keys_to_output(keys):
    """
    Convert key inputs to one-hot encoded output array
    
    Output format:
    [W, S, A, D, WA, WD, SA, SD, NO_INPUT]
    """
    output = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 9 possible outputs
    
    if 'W' in keys and 'A' in keys:
        output[4] = 1  # Forward + Left
    elif 'W' in keys and 'D' in keys:
        output[5] = 1  # Forward + Right
    elif 'S' in keys and 'A' in keys:
        output[6] = 1  # Backward + Left
    elif 'S' in keys and 'D' in keys:
        output[7] = 1  # Backward + Right
    elif 'W' in keys:
        output[0] = 1  # Forward
    elif 'S' in keys:
        output[1] = 1  # Backward
    elif 'A' in keys:
        output[2] = 1  # Left
    elif 'D' in keys:
        output[3] = 1  # Right
    else:
        output[8] = 1  # No input
    
    return output

def balance_data():
    """Balance training data to have an equal number of samples for each class"""
    print("Balancing training data...")
    import balance_data
    balance_data.balance_data()

def train_model():
    """Train the neural network model with collected data"""
    print("Training model...")
    import train_model
    train_model.train_model(width=WIDTH, height=HEIGHT, epochs=EPOCHS)

def test_model():
    """Test the trained model in self-driving mode"""
    print("Testing model in self-driving mode...")
    import test_model
    test_model.test_model(width=WIDTH, height=HEIGHT)

def evaluate_model():
    """Evaluate saved model on a held-out test split and print accuracy, precision, recall, and F1 score"""
    import numpy as np
    import os
    from nervenet import nervenet
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    # Locate latest saved model
    model_dir = 'model'
    model_entries = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    if not model_entries:
        print("No saved models found in 'model' directory.")
        return
    latest_model = sorted(model_entries)[-1]
    model_folder = os.path.join(model_dir, latest_model)
    # Construct checkpoint prefix: path to the files without extension
    ckpt_prefix = os.path.join(model_folder, latest_model)
    print(f"Loading model from checkpoint prefix '{ckpt_prefix}' for evaluation...")
    model = nervenet(WIDTH, HEIGHT, LR)
    model.load(ckpt_prefix)
    # Load screen-based balanced data for evaluation
    data_file = 'balanced_training_data_screen.npy'
    try:
        data = np.load(data_file, allow_pickle=True)
    except FileNotFoundError:
        print(f"Data file '{data_file}' not found.")
        return
    X = np.array([item[0] for item in data]).reshape(-1, HEIGHT, WIDTH, 3) / 255.0
    Y = np.array([item[1] for item in data])
    # Create train/test split
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    # Predict on test set
    predictions = model.predict({'input': X_test})
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(Y_test, axis=1)
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    # Display results
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

def detect_waypoints_only():
    """Run only the waypoint detection algorithm without neural network"""
    print("Waypoint Detection Mode")
    print("=" * 50)
    print("This mode will only detect waypoints and show the recommended steering.")
    print("No actual driving will be performed.")
    print("Controls:")
    print("  - Q: Quit")
    print("=" * 50)
    
    # Wait for user to get ready
    input("Press Enter when you're ready to start...")
    
    # Countdown
    countdown()
    
    print("Ready! Press 'Q' to quit")
    
    last_time = time.time()
    
    try:
        while True:
            # Check for key presses
            keys = key_check()
            
            # Quit if Q is pressed
            if 'Q' in keys:
                print("Quitting...")
                break
            
            # Grab screen
            screen = grab_screen(region=GAME_REGION)
            
            # Detect waypoints
            waypoints, debug_waypoints = draw_lanes.detect_waypoints(screen, debug=True)
            
            # Calculate steering based on waypoints
            waypoint_steering, debug_steering = draw_lanes.calculate_steering(screen, waypoints, debug=True)
            
            # Calculate FPS
            fps = 1 / (time.time() - last_time)
            last_time = time.time()
            
            # Display FPS
            cv2.putText(debug_steering, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the processed image
            cv2.imshow('Waypoint Detection', debug_steering)
            
            # Press ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    finally:
        # Clean up
        cv2.destroyAllWindows()

def main():
    """Main menu function"""
    while True:
        clear_screen()
        print("=" * 50)
        print("Forza Horizon 4 Self-Driving System")
        print("=" * 50)
        print("1. Collect Training Data")
        print("2. Balance Training Data")
        print("3. Train Model")
        print("4. Test Model (Self-Driving Mode)")
        print("5. Waypoint Detection Only")
        print("6. Evaluate Model")
        print("7. Quit")
        print("=" * 50)
        
        choice = input("Enter your choice (1-7): ")
        
        if choice == '1':
            collect_training_data()
        elif choice == '2':
            balance_data()
        elif choice == '3':
            train_model()
        elif choice == '4':
            test_model()
        elif choice == '5':
            detect_waypoints_only()
        elif choice == '6':
            evaluate_model()
            input("Press Enter to return to menu...")
        elif choice == '7':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    # Fix GPU memory allocation issues
    tf_fix.fix_gpu_memory_allocation()
    
    # Run the main menu
    main()