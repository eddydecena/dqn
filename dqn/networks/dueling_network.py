from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor 
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling

def build_q_network(n_actions: int, input_shape: Tuple[int]=(84, 84), history_length: int=4):
    model_input = layers.Input(shape=(input_shape[0], input_shape[1], history_length))
    x = layers.Lambda(lambda layer: layer / 255)(model_input)  # normalize by 255

    x = layers.Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = layers.Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = layers.Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = layers.Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)

    # Split into value and advantage streams
    val_stream, adv_stream = layers.Lambda(lambda w: tf.split(w, 2, 3))(x)  # custom splitting layer

    val_stream = layers.Flatten()(val_stream)
    val = layers.Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)

    adv_stream = layers.Flatten()(adv_stream)
    adv = layers.Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

    # Combine streams into Q-Values
    reduce_mean = layers.Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))  # custom layer for reduce mean

    q_vals = layers.Add()([val, layers.Subtract()([adv, reduce_mean(adv)])])

    # Build model
    model = Model(model_input, q_vals)
    # model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model

# # Read more about de dueling
# class DuelingDQN(Model):
#     """
#     This is a implementation of Dueling Network propose in https://arxiv.org/pdf/1511.06581.pdf
#     """
#     def __init__(self, n_action: int, input_shape: Tuple[int], history_length: int = 4) -> None:
#         super(DuelingDQN, self).__init__()
        
#         self._input_shape = input_shape
#         self.history_length = history_length
        
#         # Define convolutional layers
#         self.convolution = Sequential([
#             layers.Lambda(lambda inputs: inputs / 255.), # Normalize input 
#             layers.Conv2D(32, kernel_size=(8, 8), strides=4, activation='relu'),
#             layers.Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu'),
#             layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu'),
#             layers.Conv2D(1024, kernel_size=(7, 7), strides=1, activation='relu')
#         ])
        
#         # split between value_stream and advantage_stream
#         self.split = layers.Lambda(lambda w: split(w, 2, 3))
        
#         # Advantage stream forward
#         self.advantage_stream = Sequential([
#             layers.Flatten(),
#             layers.Dense(n_action, kernel_initializer=VarianceScaling(scale=2.))
#         ])
        
#         # Value stream forward
#         self.value_stream = Sequential([
#             layers.Flatten(),
#             layers.Dense(1, kernel_initializer=VarianceScaling(scale=2.))
#         ])
        
#         # Putting all together
#         self.reduce_mean = layers.Lambda(lambda w: reduce_mean(w, axis=1, keepdims=True))
#         self.subtract = layers.Subtract()
#         self.outputs = layers.Add()
        
#     def call(self, inputs: Tensor) -> Tensor:
#         x = layers.Input(shape=(self._input_shape[0], self._input_shape[1], self.history_length))(inputs)
#         x = self.convolution(x)
        
#         value_stream, advantage_stream = self.split(x)
#         value_stream = self.value_stream(value_stream)
#         advantage_stream = self.advantage_stream(advantage_stream)
#         subtracted = self.subtract([advantage_stream, self.reduce_mean(advantage_stream)])
        
#         return self.outputs([value_stream, subtracted])
        