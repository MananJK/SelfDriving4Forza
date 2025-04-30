import cv2
import numpy as np
import time
import math
from ImageGrab import grab_screen
from getkeys import key_check
import pytesseract
import re
import os
from direct_input import XboxController  # Import the XboxController class

# Tesseract path handling with fallbacks
tesseract_installed = True
try:
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'D:\Program Files\Tesseract-OCR\tesseract.exe',
        # Add other potential paths here
    ]
    
    if pytesseract.pytesseract.tesseract_cmd == 'tesseract' or not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        # Try each possible path
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"Found Tesseract at: {path}")
                break
        else:
            # If loop completes without finding Tesseract
            print("WARNING: Tesseract OCR not found. Speed detection will be disabled.")
            print("To enable speed detection, install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
            tesseract_installed = False
    else:
        # Tesseract path is already set and valid
        print(f"Using Tesseract at: {pytesseract.pytesseract.tesseract_cmd}")
except Exception as e:
    print(f"ERROR initializing Tesseract: {e}")
    print("Speed detection will be disabled.")
    tesseract_installed = False

# Constants
WINDOW_NAME = "Forza Horizon 4 Waypoint Detection"
SPEED_WINDOW_NAME = "Speedometer Monitor"

# Default regions
MINIMAP_REGION = (0, 560, 250, 720)
FULL_SCREEN_REGION = (0, 0, 1024, 768)
SPEEDOMETER_REGION = (750, 680, 850, 720)

# Default HSV color range for blue route line
LOWER_BLUE = np.array([90, 150, 150])
UPPER_BLUE = np.array([130, 255, 255])

# Default HSV color range for white/silver car arrow
LOWER_ARROW = np.array([0, 0, 150])  # Light colors (white/silver/gray)
UPPER_ARROW = np.array([180, 30, 255])

def detect_car_arrow(minimap, debug=False):
    # Create a copy for visualization
    vis_img = minimap.copy() if debug else None
    
    # Assume car is at the center of minimap (most games place player at center)
    minimap_center = (minimap.shape[1] // 2, minimap.shape[0] // 2)
    car_position = minimap_center
    
    # Default car angle (assume facing upward/north)
    car_angle = 0
    
    if debug:
        # Draw car position at center
        cv2.circle(vis_img, (int(car_position[0]), int(car_position[1])), 5, (255, 0, 0), -1)
        
        # Draw car orientation line (assuming default north orientation)
        end_x = int(car_position[0] + 20 * math.cos(math.radians(car_angle - 90)))
        end_y = int(car_position[1] + 20 * math.sin(math.radians(car_angle - 90)))
        cv2.line(vis_img, (int(car_position[0]), int(car_position[1])), (end_x, end_y), (255, 0, 255), 2)
        
        # Add orientation angle text
        cv2.putText(vis_img, f"{int(car_angle)}°", (int(car_position[0]) + 5, int(car_position[1]) - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add title to the visualization
        cv2.putText(vis_img, "Car Position (Center)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return car_position, car_angle, vis_img
    
    return car_position, car_angle, None

def extract_speed(img):
    # If tesseract is not installed, don't try OCR
    if not tesseract_installed:
        return 0
        
    try:
        # Preprocess image for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get white text on black background
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Apply some blur to remove noise
        blur = cv2.GaussianBlur(thresh, (3, 3), 0)
        
        # Run OCR
        text = pytesseract.image_to_string(blur, config='--psm 7 -c tessedit_char_whitelist=0123456789')
        
        # Extract numbers
        numbers = re.findall(r'\d+', text)
        
        if numbers:
            try:
                # Get the first number as speed
                speed = int(numbers[0])
                return speed
            except ValueError:
                return 0
        
        return 0
    except Exception as e:
        print(f"Error in OCR processing: {e}")
        return 0

def detect_waypoints(img, debug=True):
    # Convert to HSV for better color filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the color
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    
    # Noise removal with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract centers of waypoints
    waypoints = []
    for contour in contours:
        # Filter out small contours (noise)
        if cv2.contourArea(contour) > 10:  # Small threshold for minimap
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                waypoints.append((cx, cy))
    
    # For debugging, draw the waypoints on the image
    if debug:
        debug_img = img.copy()
        
        # Draw the mask in green with transparency
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored[np.where((mask_colored == [255, 255, 255]).all(axis=2))] = [0, 255, 0]
        debug_img = cv2.addWeighted(debug_img, 0.7, mask_colored, 0.3, 0)
        
        # Draw circles at waypoint centers
        for point in waypoints:
            cv2.circle(debug_img, point, 3, (0, 0, 255), -1)
        
        # Draw car position (center of the minimap)
        height, width = img.shape[:2]
        car_pos = (width // 2, height // 2)
        cv2.circle(debug_img, car_pos, 5, (255, 0, 0), -1)
        
        # If we have waypoints, draw line to closest one and show direction
        if waypoints:
            # Sort waypoints by distance from car
            waypoints_dist = [(wp, np.sqrt((wp[0] - car_pos[0])**2 + (wp[1] - car_pos[1])**2)) 
                             for wp in waypoints]
            waypoints_dist.sort(key=lambda x: x[1])
            
            # Get closest waypoint
            closest = waypoints_dist[0][0]
            
            # Draw line to closest waypoint
            cv2.line(debug_img, car_pos, closest, (0, 255, 0), 2)
            
            # Calculate steering direction
            dx = closest[0] - car_pos[0]
            dy = car_pos[1] - closest[1]  # Inverted Y-axis
            
            angle = np.degrees(np.arctan2(dx, dy))
            
            # Determine direction
            if -20 <= angle <= 20:
                direction = "Straight"
            elif 20 < angle <= 150:
                direction = "Right"
            elif angle > 150 or angle < -150:
                direction = "Backward"
            elif -150 <= angle < -20:
                direction = "Left"
            
            # Show direction text
            cv2.putText(debug_img, f"Dir: {direction}", (5, 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return waypoints, debug_img
    
    return waypoints, None

def suggest_direction(minimap, waypoints, car_position, car_angle):
    debug_img = minimap.copy()
    
    # Default direction and steering
    direction = "Unknown"
    steering_value = 0.0
    
    # If no waypoints detected
    if not waypoints:
        cv2.putText(debug_img, "No waypoints", (5, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return direction, steering_value, debug_img
    
    # Convert car position to integers for display
    car_pos_int = (int(car_position[0]), int(car_position[1]))
    
    # Calculate distances to all waypoints
    waypoints_dist = [(wp, np.sqrt((wp[0] - car_pos_int[0])**2 + (wp[1] - car_pos_int[1])**2)) 
                     for wp in waypoints]
    
    # Sort by distance
    waypoints_dist.sort(key=lambda x: x[1])
    
    # Get closest waypoint
    closest_wp = waypoints_dist[0][0]
    
    # Draw car position
    cv2.circle(debug_img, car_pos_int, 5, (255, 0, 0), -1)
    
    # Draw closest waypoint
    cv2.circle(debug_img, closest_wp, 5, (0, 255, 0), -1)
    
    # Draw line from car to closest waypoint
    cv2.line(debug_img, car_pos_int, closest_wp, (0, 255, 255), 2)
    
    # Calculate waypoint direction relative to car position
    wp_dx = closest_wp[0] - car_pos_int[0]
    wp_dy = closest_wp[1] - car_pos_int[1]

    # Compute angle to waypoint: 0° = straight ahead (north), positive = turn right, negative = turn left
    # Compute angle to waypoint relative to car forward vector
    angle_to_wp = math.degrees(math.atan2(wp_dx, -wp_dy))  # -180 to 180, 0 = straight ahead
    # Normalize relative angle to [-180, 180]
    relative_angle = angle_to_wp - car_angle
    if relative_angle > 180:
        relative_angle -= 360
    elif relative_angle < -180:
        relative_angle += 360
    # For display, normalize waypoint angle to [0,360)
    wp_angle = angle_to_wp % 360

    # Calculate distance from car to waypoint (for intensity calculation)
    distance = math.sqrt(wp_dx**2 + wp_dy**2)
    max_distance = math.sqrt(minimap.shape[0]**2 + minimap.shape[1]**2) / 2
    
    # Calculate normalized distance (0.0 to 1.0)
    normalized_distance = min(1.0, distance / max_distance)
    
    # Draw quadrant lines (for visualization)
    height, width = minimap.shape[:2]
    cv2.line(debug_img, (car_pos_int[0], 0), (car_pos_int[0], height), (100, 100, 100), 1)
    cv2.line(debug_img, (0, car_pos_int[1]), (width, car_pos_int[1]), (100, 100, 100), 1)
    
    # Determine quadrant and steering intensity based on relative angle
    STRAIGHT_THRESHOLD = 10.0  # degrees
    TURN_AROUND_THRESHOLD = 170.0  # degrees
    if abs(relative_angle) <= STRAIGHT_THRESHOLD:
        direction = "Straight"
        steering_value = relative_angle / 180.0  # small gentle steering
    elif abs(relative_angle) < TURN_AROUND_THRESHOLD:
        if relative_angle > 0:
            direction = "Right"
            intensity = (relative_angle - STRAIGHT_THRESHOLD) / (TURN_AROUND_THRESHOLD - STRAIGHT_THRESHOLD)
            steering_value = 0.5 + (intensity * 0.5)
        else:
            direction = "Left"
            intensity = (abs(relative_angle) - STRAIGHT_THRESHOLD) / (TURN_AROUND_THRESHOLD - STRAIGHT_THRESHOLD)
            steering_value = -0.5 - (intensity * 0.5)
    else:
        direction = "Turn Around"
        steering_value = 1.0 if relative_angle > 0 else -1.0

    # Add a small multiplier based on distance - farther waypoints need less aggressive steering
    distance_factor = 1.0 - (normalized_distance * 0.5)  # Ranges from 0.5 to 1.0
    steering_value *= distance_factor
    
    # Draw quadrant label
    quadrant_text = ""
    if wp_dx >= 0 and wp_dy < 0:
        quadrant_text = "Q1: Upper Right"
    elif wp_dx < 0 and wp_dy < 0:
        quadrant_text = "Q2: Upper Left"
    elif wp_dx < 0 and wp_dy >= 0:
        quadrant_text = "Q3: Lower Left"
    else:
        quadrant_text = "Q4: Lower Right"
    
    # Draw debug info
    cv2.putText(debug_img, f"Dir: {direction}", (5, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(debug_img, f"Steer: {steering_value:.2f}", (5, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(debug_img, quadrant_text, (5, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Draw car angle
    end_x = int(car_pos_int[0] + 20 * math.cos(math.radians(car_angle - 90)))
    end_y = int(car_pos_int[1] + 20 * math.sin(math.radians(car_angle - 90)))
    cv2.line(debug_img, car_pos_int, (end_x, end_y), (255, 0, 255), 2)
    
    # Draw waypoint angle
    end_x = int(car_pos_int[0] + 20 * math.cos(math.radians(wp_angle - 90)))
    end_y = int(car_pos_int[1] + 20 * math.sin(math.radians(wp_angle - 90)))
    cv2.line(debug_img, car_pos_int, (end_x, end_y), (0, 255, 0), 2)
    
    # Add more debug text
    cv2.putText(debug_img, f"Car: {int(car_angle)}°", (5, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    cv2.putText(debug_img, f"WP: {int(wp_angle)}°", (5, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(debug_img, f"Rel: {int(relative_angle)}°", (5, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return direction, steering_value, debug_img

def main():
    last_time = time.time()
    fps_counter = 0
    fps = 0
    start_time = time.time()
    
    # Initialize the Xbox controller for steering control
    controller = XboxController()
    auto_steering = False  # Flag to toggle automatic steering
    # Burst steering control to prevent overshoot
    steering_active = False
    steer_start_time = 0.0
    BURST_DURATION = 0.5  # seconds
    current_steer_value = 0.0

    # Allow user to adjust minimap position with keyboard
    minimap_x, minimap_y = 0, 560  # Initial position - bottom left area
    minimap_size = 250  # Size of the minimap
    
    print("Starting detection. Press 'q' to quit.")
    print("Use arrow keys to adjust minimap position if needed.")
    print("Press 'R' to toggle automatic steering")
    
    while True:
        # Capture screen
        screen = grab_screen()
        
        # Extract minimap region
        minimap = screen[minimap_y:minimap_y+minimap_size, minimap_x:minimap_x+minimap_size]
        
        # Check if minimap is valid
        if minimap.size == 0:
            print("Warning: Minimap region is outside screen bounds!")
            # Use safer default values
            minimap_y = max(0, screen.shape[0] - 250)
            minimap_x = 0
            minimap = screen[minimap_y:minimap_y+minimap_size, minimap_x:minimap_x+minimap_size]
        
        # Get speed
        speed_img = screen[680:720, 750:850]  # Adjust these coordinates based on your game
        speed_value = extract_speed(speed_img)
        
        # Detect car arrow position and orientation (before waypoints for visualization)
        car_position, car_angle, car_debug_img = detect_car_arrow(minimap, True)
        
        # Detect waypoints using blue color
        waypoints, waypoint_img = detect_waypoints(minimap, True)
        
        # Get suggested direction based on waypoints and car position
        direction, steering_value, direction_img = suggest_direction(minimap, waypoints, car_position, car_angle)
        
        # Apply steering based on the direction if auto-steering is enabled
        if auto_steering:
            if not steering_active:
                if direction in ["Left", "Right", "Turn Around"]:
                    steering_active = True
                    steer_start_time = time.time()
                    current_steer_value = steering_value
                    controller.set_steering(current_steer_value)
            else:
                if time.time() - steer_start_time > BURST_DURATION:
                    steering_active = False
                    controller.set_steering(0.0)
        else:
            # Release steering control when auto-steering is disabled
            controller.set_steering(0.0)
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - start_time > 1:
            fps = fps_counter
            fps_counter = 0
            start_time = time.time()
        
        # Create visualization - only using the left window (waypoint_img)
        if waypoint_img is not None:
            # Use only the waypoint_img 
            vis_minimap = waypoint_img.copy()
            
            # Add text overlay with information
            cv2.putText(vis_minimap, f"Speed: {speed_value} km/h", (5, 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add OCR status indicator
            if not tesseract_installed:
                cv2.putText(vis_minimap, "OCR Disabled (No Tesseract)", (5, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.putText(vis_minimap, f"FPS: {fps}", (5, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_minimap, f"Waypoints: {len(waypoints)}", (5, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_minimap, f"Car angle: {car_angle:.1f}°", (5, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_minimap, f"Direction: {direction}", (5, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Make direction more visible
            cv2.putText(vis_minimap, f"Minimap pos: ({minimap_x}, {minimap_y})", (5, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_minimap, f"Auto-steering: {'ON' if auto_steering else 'OFF'}", (5, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Display the visualization
            cv2.imshow("Minimap Analysis", vis_minimap)
        
        print(f'FPS: {fps}, Speed: {speed_value} km/h, Direction: {direction}, Car angle: {car_angle:.1f}°, Auto-steering: {"ON" if auto_steering else "OFF"}')
        
        # Check for key presses to adjust minimap position and toggle auto-steering
        keys = key_check()
        if 'Q' in keys:
            cv2.destroyAllWindows()
            break
        if 'UP' in keys:
            minimap_y = max(0, minimap_y - 10)
        if 'DOWN' in keys:
            minimap_y = min(screen.shape[0] - minimap_size, minimap_y + 10)
        if 'LEFT' in keys:
            minimap_x = max(0, minimap_x - 10)
        if 'RIGHT' in keys:
            minimap_x = min(screen.shape[1] - minimap_size, minimap_x + 10)
        if 'R' in keys:
            # Wait for key release to avoid multiple toggles
            while 'R' in key_check():
                time.sleep(0.1)
            auto_steering = not auto_steering
            print(f"Auto-steering: {'ON' if auto_steering else 'OFF'}")
        
        # Delay for visualization
        cv2.waitKey(1)

if __name__ == "__main__":
    main()