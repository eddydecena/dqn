from typing import Tuple

from tensorflow import split
from tensorflow import reduce_mean
from tensorflow import Tensor 
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.initializers import VarianceScaling

# Read more about de dueling
class DuelingDQN(Model):
    """
    This is a implementation of Dueling Network propose in https://arxiv.org/pdf/1511.06581.pdf
    """
    def __init__(self, n_action: int, input_shape: Tuple[int], history_length: int = 4) -> None:
        super(DuelingDQN, self).__init__()
        
        # Define convolutional layers
        self.convolution = Sequential([
            layers.Input(shape=(input_shape[0], input_shape[1], history_length)),
            layers.Lambda(lambda inputs: inputs / 255.), # Normalize input 
            layers.Conv2D(32, kernel_size=(8, 8), strides=4, activation='relu'),
            layers.Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu'),
            layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu'),
            layers.Conv2D(1024, kernel_size=(7, 7), strides=1, activation='relu')
        ])
        
        # split between value_stream and advantage_stream
        self.split = layers.Lambda(lambda w: split(w, 2, 3))
        
        # Advantage stream forward
        self.advantage_stream = Sequential([
            layers.Flatten(),
            layers.Dense(n_action, kernel_initializer=VarianceScaling(scale=2.))
        ])
        
        # Value stream forward
        self.value_stream = Sequential([
            layers.Flatten(),
            layers.Dense(1, kernel_initializer=VarianceScaling(scale=2.))
        ])
        
        # Putting all together
        self.reduce_mean = layers.Lambda(lambda w: reduce_mean(w, axis=1, keepdims=True))
        self.subtract = layers.Subtract()
        self.outputs = layers.Add()
        
    def call(self, inputs: Tensor) -> Tensor:
        x = self.convolution(inputs)
        
        value_stream, advantage_stream = self.split(x)
        value_stream = self.value_stream(value_stream)
        advantage_stream = self.advantage_stream(advantage_stream)
        subtracted = self.subtract([advantage_stream, self.reduce_mean(advantage_stream)])
        
        return self.outputs([value_stream, subtracted])
        