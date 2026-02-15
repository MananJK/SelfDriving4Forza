import numpy as np
import cv2
import time
import os
import tensorflow as tf
from nervenet import nervenet
import tf_fix
from draw_lanes import detect_waypoints, calculate_steering
from ImageGrab import grab_screen
import direct_input
import getkeys
import random

def test_model(model_path=None, width=160, height=120):
    # Look for models if none specified
    if model_path is None:
        model_paths = [f for f in os.listdir('model') if not f.startswith('.')]
        if not model_paths:
            print("No models found in the 'model' directory. Train a model first.")
            return
        model_path = os.path.join('model', model_paths[-1])  # Use the most recent model
    
    print(f"Loading model from {model_path}...")
    
    # Create the model
    lr = 1e-3  # Learning rate (doesn't matter for inference)
    model = nervenet(width, height, lr)
    
    # Load trained weights
    model.load(model_path)
    print("Model loaded!")
    
    # Define game screen region for 1024x768 resolution
    game_region = (0, 40, 1024, 768)  # Updated game region
    
    print("Starting self-driving mode. Press 'Q' to quit, 'P' to pause, 'R' to change quadrants.")
    
    # Countdown before starting
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    # Main loop
    last_time = time.time()
    paused = False
    
    # Initialize direct input controller
    controller = direct_input.XboxController()
    # Burst steering control to prevent overshoot
    steering_active = False
    steer_start_time = 0.0
    BURST_DURATION = 0.5  # seconds
    current_steer_value = 0.0
    
    # Initialize quadrant variables
    quadrants = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
    current_quadrant = "Top-Left"  # Default quadrant
    quadrant_positions = {
        "Top-Left": (0, 0),
        "Top-Right": (game_region[2]//2, 0),
        "Bottom-Left": (0, game_region[3]//2),
        "Bottom-Right": (game_region[2]//2, game_region[3]//2)
    }
    quadrant_dimensions = (game_region[2]//2, game_region[3]//2)
    
    try:
        while True:
            # Check for key presses
            keys = getkeys.key_check()
            
            # Quit if Q is pressed
            if 'Q' in keys:
                print("Quitting...")
                break
                
            # Pause/unpause if P is pressed
            if 'P' in keys:
                paused = not paused
                print("Paused" if paused else "Unpaused")
                time.sleep(0.5)  # Add delay to avoid multiple toggles
            
            # Change quadrant if R is pressed
            if 'R' in keys:
                # Select a different quadrant randomly
                new_quadrant = current_quadrant
                while new_quadrant == current_quadrant:
                    new_quadrant = random.choice(quadrants)
                current_quadrant = new_quadrant
                print(f"Switched to {current_quadrant} quadrant")
                time.sleep(0.5)  # Add delay to avoid multiple toggles
            
            if not paused:
                # Grab screen
                screen = grab_screen(region=game_region)
                
                # Resize for faster processing
                screen_resized = cv2.resize(screen, (width, height))
                
                # Detect waypoints
                waypoints, debug_waypoints = detect_waypoints(screen, debug=True)
                
                # Calculate steering based on waypoints (rule-based approach)
                waypoint_steering, debug_steering = calculate_steering(screen, waypoints, debug=True)
                
                # Process the image for neural network input
                screen_processed = cv2.resize(screen, (width, height))
                screen_processed = screen_processed.reshape(-1, height, width, 3) / 255.0
                
                # Predict with the model
                prediction = model.predict({'input': screen_processed})[0]
                action = np.argmax(prediction)
                
                # Map the predicted action to controller inputs
                # [Forward, Backward, Left, Right, Forward+Left, Forward+Right, Backward+Left, Backward+Right, No Input]
                controls = {
                    0: {'acceleration': 1.0, 'brake': 0.0, 'steering': 0.0},  # Forward
                    1: {'acceleration': 0.0, 'brake': 1.0, 'steering': 0.0},  # Backward
                    2: {'acceleration': 0.0, 'brake': 0.0, 'steering': -1.0},  # Left
                    3: {'acceleration': 0.0, 'brake': 0.0, 'steering': 1.0},  # Right
                    4: {'acceleration': 1.0, 'brake': 0.0, 'steering': -1.0},  # Forward+Left
                    5: {'acceleration': 1.0, 'brake': 0.0, 'steering': 1.0},  # Forward+Right
                    6: {'acceleration': 0.0, 'brake': 1.0, 'steering': -1.0},  # Backward+Left
                    7: {'acceleration': 0.0, 'brake': 1.0, 'steering': 1.0},  # Backward+Right
                    8: {'acceleration': 0.0, 'brake': 0.0, 'steering': 0.0}   # No Input
                }
                
                # Get controls for the predicted action
                control = controls[action]
                
                # Apply steering with burst control
                steer_val = control['steering']
                if not steering_active:
                    # Start burst if non-zero steering requested
                    if abs(steer_val) > 0.2:
                        steering_active = True
                        steer_start_time = time.time()
                        current_steer_value = steer_val
                        controller.set_steering(current_steer_value)
                else:
                    # End burst after duration
                    if time.time() - steer_start_time > BURST_DURATION:
                        steering_active = False
                        controller.set_steering(0.0)
                # Apply throttle and brake directly
                controller.set_throttle(control['acceleration'])
                controller.set_brake(control['brake'])
                
                # Calculate and display FPS
                fps = 1 / (time.time() - last_time)
                last_time = time.time()
                
                # Display info
                cv2.putText(debug_steering, f"FPS: {fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_steering, f"Action: {action}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_steering, f"Quadrant: {current_quadrant}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw quadrant overlay
                quadrant_display = debug_steering.copy()
                h, w = quadrant_display.shape[:2]
                # Draw horizontal line
                cv2.line(quadrant_display, (0, h//2), (w, h//2), (255, 255, 255), 2)
                # Draw vertical line
                cv2.line(quadrant_display, (w//2, 0), (w//2, h), (255, 255, 255), 2)

                # Add quadrant labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                color = (255, 255, 255)
                thickness = 2
                
                # Top-Left quadrant
                cv2.putText(quadrant_display, "Top-Left", (10, 30), font, font_scale, color, thickness)
                # Top-Right quadrant
                cv2.putText(quadrant_display, "Top-Right", (w//2 + 10, 30), font, font_scale, color, thickness)
                # Bottom-Left quadrant
                cv2.putText(quadrant_display, "Bottom-Left", (10, h//2 + 30), font, font_scale, color, thickness)
                # Bottom-Right quadrant
                cv2.putText(quadrant_display, "Bottom-Right", (w//2 + 10, h//2 + 30), font, font_scale, color, thickness)
                
                # Highlight current quadrant
                if current_quadrant == "Top-Left":
                    cv2.rectangle(quadrant_display, (0, 0), (w//2-1, h//2-1), (0, 255, 0), 3)
                elif current_quadrant == "Top-Right":
                    cv2.rectangle(quadrant_display, (w//2, 0), (w-1, h//2-1), (0, 255, 0), 3)
                elif current_quadrant == "Bottom-Left":
                    cv2.rectangle(quadrant_display, (0, h//2), (w//2-1, h-1), (0, 255, 0), 3)
                elif current_quadrant == "Bottom-Right":
                    cv2.rectangle(quadrant_display, (w//2, h//2), (w-1, h-1), (0, 255, 0), 3)

                # Display the quadrant grid
                cv2.imshow('Self-Driving Visualization', quadrant_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    finally:
        # Clean up
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    test_model()