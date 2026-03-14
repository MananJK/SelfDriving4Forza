import numpy as np
import tensorflow as tf
from nervenet import nervenet
import tf_fix
import os
import time

def train_model(file_path='balanced_training_data_screen.npy', width=160, height=120, epochs=50, batch_size=32, save_model=True):
    """
    Trains the neural network model using collected and balanced training data
    
    Args:
        file_path: Path to the balanced training data file
        width: Input image width
        height: Input image height
        epochs: Number of training epochs
        batch_size: Training batch size
        save_model: Whether to save the model after training
    
    Returns:
        Trained model
    """
    # Load the training data
    print(f"Loading training data from {file_path}...")
    try:
        train_data = np.load(file_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"File {file_path} not found. Make sure you have balanced data first.")
        return None
    
    print(f"Training data shape: {train_data.shape}")
    
    # Prepare the training data
    train_x = np.array([i[0] for i in train_data]).reshape(-1, height, width, 3)
    train_y = np.array([i[1] for i in train_data])
    
    # Normalize pixel values to be between 0 and 1
    train_x = train_x / 255.0
    
    # Create a model
    print("Creating model...")
    lr = 1e-3  # Learning rate
    model = nervenet(width, height, lr)
    
    # Create directory for model if it doesn't exist
    if not os.path.exists('model'):
        os.makedirs('model')
    
    # Define model path
    model_name = f'forza_waypoint_driver-{int(time.time())}'
    model_path = f'model/{model_name}'
    
    # Train the model
    print(f"Training model with {len(train_x)} samples...")
    
    # Create a TensorBoard callback
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{model_name}')
    
    model.fit({'input': train_x}, {'targets': train_y}, 
              n_epoch=epochs,
              validation_set=0.1,
              show_metric=True, 
              batch_size=batch_size,
              shuffle=True)
    
    # Save the model
    if save_model:
        print(f"Saving model to {model_path}...")
        model.save(model_path)
        print(f"Model saved!")
    
    return model

if __name__ == "__main__":
    train_model()