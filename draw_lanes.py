import cv2
import numpy as np

def enhance_road_markings(img):
    """
    Enhance the visibility of road markings using color manipulation
    
    Args:
        img: Input image (screenshot)
        
    Returns:
        enhanced_img: Image with enhanced road markings
    """
    # Create a copy of the original image
    enhanced_img = img.copy()
    
    # Method 1: Increase contrast
    enhanced_img = cv2.convertScaleAbs(enhanced_img, alpha=1.5, beta=0)
    
    # Method 2: Apply color boost to blue channel
    b, g, r = cv2.split(enhanced_img)
    b = cv2.convertScaleAbs(b, alpha=1.3, beta=10)  # Boost blue channel
    enhanced_img = cv2.merge((b, g, r))
    
    # Method 3: Optional color inversion for testing
    # enhanced_img = cv2.bitwise_not(enhanced_img)
    
    return enhanced_img

def detect_blue_line(img, debug=False, enhance=True):
    """
    Detect the blue racing line in the game by color thresholding
    
    Args:
        img: Input image (screenshot)
        debug: If True, returns an image with detected line highlighted
        enhance: If True, preprocess the image to enhance road markings
        
    Returns:
        line_points: List of points along the blue line
        center_line: Center points of the blue line
        debug_img: Image with line highlighted (if debug=True)
    """
    # Apply enhancement if requested
    if enhance:
        img = enhance_road_markings(img)
    
    # Convert to HSV for better color filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for blue line color
    # These values may need adjustment based on the actual blue line color
    lower_blue = np.array([100, 150, 150])
    upper_blue = np.array([140, 255, 255])
    
    # Create a mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Noise removal with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find the blue line points
    line_points = np.column_stack(np.where(mask > 0))
    if len(line_points) == 0:
        return [], [], img if debug else None
    
    # Convert coordinate format from [y, x] to [x, y] for easier processing
    line_points = np.flip(line_points, axis=1)
    
    # Sort points by Y coordinate (from bottom to top of image)
    line_points = line_points[line_points[:, 1].argsort()]
    
    # Extract center line by segmenting the line horizontally and finding centers
    height, width = img.shape[:2]
    num_segments = 20  # Number of horizontal segments to divide the image
    segment_height = height // num_segments
    
    center_line = []
    for i in range(num_segments):
        y_min = height - (i + 1) * segment_height
        y_max = height - i * segment_height
        
        # Get points in this segment
        segment_points = line_points[(line_points[:, 1] >= y_min) & (line_points[:, 1] < y_max)]
        
        if len(segment_points) > 0:
            # Find average x-coordinate in this segment
            avg_x = np.mean(segment_points[:, 0]).astype(int)
            center_line.append((avg_x, y_min + segment_height // 2))
    
    # For debugging, draw the blue line on the image
    if debug:
        debug_img = img.copy()
        
        # Draw the mask in green with transparency
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored[np.where((mask_colored == [255, 255, 255]).all(axis=2))] = [0, 255, 0]
        debug_img = cv2.addWeighted(debug_img, 0.7, mask_colored, 0.3, 0)
        
        # Draw the center line
        for i in range(1, len(center_line)):
            cv2.line(debug_img, center_line[i-1], center_line[i], (0, 0, 255), 2)
        
        # Draw circles at center points
        for point in center_line:
            cv2.circle(debug_img, point, 3, (255, 0, 0), -1)
            
        return line_points, center_line, debug_img
    
    return line_points, center_line, None

def calculate_steering_from_line(img, center_line, debug=False):
    """
    Calculate steering angle based on the detected blue line
    
    Args:
        img: Input image
        center_line: List of center points of the blue line
        debug: If True, returns an image with steering visualization
        
    Returns:
        steering: Steering value (-1 to 1, where -1 is full left, 1 is full right)
        debug_img: Image with steering visualization (if debug=True)
    """
    if not center_line or len(center_line) < 2:
        return 0, img if debug else None
    
    height, width = img.shape[:2]
    
    # Define car position (bottom center of the screen)
    car_pos = (width // 2, height - 30)
    
    # Simple steering calculation using a target point ahead
    # Use a point that's a few segments ahead (about 1/3 up the screen)
    target_index = min(len(center_line) - 1, 5)
    target_point = center_line[target_index]
    
    # Calculate offset from center
    offset = target_point[0] - (width // 2)
    
    # Normalize the offset to get steering value between -1 and 1
    max_offset = width // 3  # Maximum reasonable offset
    steering = offset / max_offset
    
    # Clamp steering value to [-1, 1]
    steering = max(-1.0, min(1.0, steering))
    
    if debug:
        debug_img = img.copy()
        
        # Draw car position
        cv2.circle(debug_img, car_pos, 10, (255, 0, 0), -1)
        
        # Draw the center line
        for i in range(1, len(center_line)):
            cv2.line(debug_img, center_line[i-1], center_line[i], (0, 0, 255), 2)
        
        # Draw target point
        cv2.circle(debug_img, target_point, 8, (0, 255, 255), -1)
        
        # Draw line from car to target
        cv2.line(debug_img, car_pos, target_point, (255, 255, 0), 2)
        
        # Display steering value
        steer_text = f"Steering: {steering:.2f}"
        cv2.putText(debug_img, steer_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return steering, debug_img
    
    return steering, None

# Keep the original functions for backward compatibility
def detect_waypoints(img, debug=False):
    """Legacy function for backward compatibility"""
    line_points, center_line, debug_img = detect_blue_line(img, debug)
    if debug:
        return center_line, debug_img
    return center_line, None

def calculate_steering(img, waypoints, debug=False):
    """Legacy function for backward compatibility"""
    return calculate_steering_from_line(img, waypoints, debug)