import os
import json
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow import GradientTape
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from dqn.replay_memory import ReplayMemory
from dqn.networks.dueling_network import build_q_network

class DQNAgent():
    _save_main_q_file = '/main_q.model'
    _save_target_q_file = '/target_q.model'
    _save_meta_file = '/meta.json'
    
    def __init__(
        self, 
        # main_q: Model, 
        # target_q: Model,
        # replay_memory: ReplayMemory,
        n_actions: int,
        input_shape: Tuple = (84, 84),
        batch_size: int=32,
        history_length: int=4,
        learning_rate: float=0.00001,
        eps_initial: int=1,
        eps_final: float=0.1,
        eps_final_frame: float=0.0,
        eps_evaluation: float=0.0,
        eps_annealing_frames: int=1000000,
        replay_buffer_size: int = 1000000,
        replay_buffer_start_size: int=50000,
        max_frames: int=25000000,
        use_per: bool=True) -> None:

        self.n_actions = n_actions
        self.input_shape = input_shape
        self.history_length = history_length
        self.learning_rate = learning_rate
        self.replay_buffer_start_size = replay_buffer_start_size
        self.max_frames = max_frames
        self.batch_size = batch_size
        # self.replay_buffer = replay_memory
        self.use_per = use_per
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.replay_buffer_size = replay_buffer_size
        
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope*self.replay_buffer_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (self.max_frames - self.eps_annealing_frames - self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_frames
        
        self.replay_buffer: ReplayMemory = ReplayMemory(
                                                size=self.replay_buffer_size,
                                                input_shape=self.input_shape,
                                                history_length=self.history_length,
                                                use_per=self.use_per)
        
        # self.main_q: Model = DuelingDQN(self.n_actions, self.input_shape, self.history_length)
        # self.target_q: Model = DuelingDQN(self.n_actions, self.input_shape, self.history_length)
        
        # self.main_q.build((self.input_shape[0], self.input_shape[1], self.history_length))
        # self.target_q.build((self.input_shape[0], self.input_shape[1], self.history_length))
        
        self.main_q = build_q_network(self.n_actions, self.input_shape, self.history_length)
        self.target_q = build_q_network(self.n_actions, self.input_shape, self.history_length)
        
        self.main_q.compile(optimizer=Adam(self.learning_rate), loss=Huber())
        self.target_q.compile(optimizer=Adam(self.learning_rate), loss=Huber())
    
    def calc_epsilon(self, frame_number: int, evaluation=False) -> float:
        if evaluation:
            return self.eps_evaluation
        elif frame_number < self.replay_buffer_start_size:
            return self.eps_initial
        elif frame_number >= self.replay_buffer_start_size and frame_number < self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope_2 * frame_number + self.intercept_2

    def get_action(self, state: np.ndarray, frame_number: int, evaluation: bool=False) -> int:
        eps = self.calc_epsilon(frame_number, evaluation)
        
        if np.random.rand() < eps:
            return np.random.randint(0, self.n_actions)
        
        q_value = self.main_q.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)))[0]
        
        return q_value.argmax()
    
    def get_intermediate_representation():
        # Maybe
        pass
    
    def update_target_network(self) -> None:
        self.target_q.set_weights(self.main_q.get_weights())
    
    def add_experience(self, action: int, reward: float, frame: np.ndarray, terminal: bool, clip_reward=True) -> None:
        self.replay_buffer.add(action, reward, frame, terminal, clip_reward)
    
    def learn(self, gamma: float, frame_number: int, priority_scale: float=1.0) -> Tuple[float, np.ndarray]:
        if self.use_per:
            (states, actions, rewards, new_states, terminal), importance, indices = self.replay_buffer.get_minibatch(self.batch_size, priority_scale=priority_scale)
            importance = importance ** (1-self.calc_epsilon(frame_number))
        else:
            states, actions, rewards, new_states, terminal = self.replay_buffer.get_minibatch(self.batch_size, priority_scale=priority_scale)
        
        main_q_values = self.main_q.predict(new_states).argmax(axis=1)
        
        future_q_values = self.target_q.predict(new_states)
        double_q = future_q_values[range(self.batch_size), main_q_values] # bad use of range
        
        target_q = rewards + (gamma * double_q * (1 - terminal)) # error with terminal
        
        with GradientTape() as tape:
            q_values = self.main_q(states)
            
            #
            one_hot_actions = to_categorical(actions, self.n_actions, dtype=np.float32)
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)
            
            error = Q - target_q
            loss = Huber()(target_q, Q)
            
            if self.use_per:
                loss = tf.reduce_mean(loss * importance)

        gradients = tape.gradient(loss, self.main_q.trainable_variables)
        self.main_q.optimizer.apply_gradients(zip(gradients, self.main_q.trainable_variables))
        
        if self.use_per:
            self.replay_buffer.set_priorities(indices, error)
        
        return float(loss.numpy()), error

    def save(self, folder_name, **kwargs) -> None:
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        
        self.main_q.save(folder_name + self._save_main_q_file)
        self.target_q.save(folder_name + self._save_target_q_file)
        
        self.replay_buffer.save(folder_name)
        
        with open(folder_name + self._save_meta_file, 'w+') as f:
            f.write(json.dumps({**{'buff_count': self.replay_buffer.count, 'buff_curr': self.replay_buffer.current}, **kwargs}))
    
    def load(self, folder_name, load_replay_buffer=True) -> dict:
        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} do not exists or is not a directory')
        
        self.main_q = load_model(folder_name + self._save_main_q_file)
        self.target_q = load_model(folder_name + self._save_target_q_file)
        self.optimizer = self.main_q.optimizer # What?
        
        if load_replay_buffer:
            self.replay_buffer.load(folder_name)
        
        with open(folder_name + '/meta.json', 'r') as f:
            meta = json.load(f)
        
        if load_replay_buffer:
            self.replay_buffer.count = meta['buff_count']
            self.replay_buffer.current = meta['buff_curr']
        
        del meta['buff_count'], meta['buff_curr']
        
        return meta