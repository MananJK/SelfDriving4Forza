import tensorflow as tf
import tensorflow.python.util.nest as util_nest
from collections.abc import Sequence
# Monkey-patch missing is_sequence for tflearn compatibility using Sequence check
util_nest.is_sequence = lambda x: isinstance(x, Sequence)

# Monkey-patch PIL.Image.ANTIALIAS for deprecated constant in tflearn datasets
from PIL import Image as PILImage
if not hasattr(PILImage, 'ANTIALIAS') and hasattr(PILImage, 'Resampling'):
    PILImage.ANTIALIAS = PILImage.Resampling.LANCZOS

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

def nervenet(width, height, lr):

    network = input_data(shape=[None, height, width, 3], name='input')
    
    # Conv Layer 1
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    
    # Conv Layer 2
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    
    # Conv Layer 3
    network = conv_2d(network, 384, 3, activation='relu')
    
    # Conv Layer 4
    network = conv_2d(network, 384, 3, activation='relu')
    
    # Conv Layer 5
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    
    # Fully Connected Layers
    network = fully_connected(network, 4096, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='relu')
    network = dropout(network, 0.5)
    
    # Output layer - 9 outputs for different driving actions:
    # [W, S, A, D, WA, WD, SA, SD, NO_INPUT]
    network = fully_connected(network, 9, activation='softmax')
    
    # Define optimizer and loss function
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')
    
    # Create model
    model = tflearn.DNN(network, checkpoint_path='model_nervenet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')
    
    return model