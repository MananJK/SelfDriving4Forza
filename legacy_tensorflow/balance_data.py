import numpy as np
import pandas as pd
from collections import Counter
import random
import os

def balance_data(file_path='training_data_screen.npy'):
    print(f"Loading data from {file_path}...")
    try:
        train_data = np.load(file_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"File {file_path} not found. Make sure you have collected training data first.")
        return None
    
    print(f"Original data shape: {train_data.shape}")
    
    # Count the occurrences of each output class
    df = pd.DataFrame(train_data)
    counter = Counter(tuple(map(tuple, df[1].values)))
    
    # Print current distribution
    print("Current data distribution:")
    for key, count in counter.items():
        keys = np.array(key).astype(int)
        action = np.argmax(keys)
        
        action_name = {
            0: "Forward (W)",
            1: "Backward (S)",
            2: "Left (A)",
            3: "Right (D)",
            4: "Forward+Left (WA)",
            5: "Forward+Right (WD)",
            6: "Backward+Left (SA)",
            7: "Backward+Right (SD)",
            8: "No Input"
        }.get(action, "Unknown")
        
        print(f"{action_name}: {count} samples")
    
    # Find the minimum count (we'll limit all classes to this amount)
    min_count = min(counter.values())
    print(f"Limiting all classes to {min_count} samples for balance")
    
    # Create balanced dataset
    balanced_data = []
    for output_type, count in counter.items():
        # Get all samples of this class
        class_data = [data for data in train_data if np.array_equal(data[1], np.array(output_type))]
        
        # Randomly select min_count samples
        selected_data = random.sample(class_data, min(min_count, len(class_data)))
        balanced_data.extend(selected_data)
    
    # Shuffle the balanced data
    random.shuffle(balanced_data)
    balanced_data = np.array(balanced_data)
    
    print(f"Balanced data shape: {balanced_data.shape}")
    
    # Save balanced data
    balanced_file_path = 'balanced_' + file_path
    np.save(balanced_file_path, balanced_data)
    print(f"Balanced data saved to {balanced_file_path}")
    
    return balanced_data

if __name__ == "__main__":
    balance_data()