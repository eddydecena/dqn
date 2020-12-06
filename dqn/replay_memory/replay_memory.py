import os
import random
from typing import Tuple

import numpy as np
from numpy.core.numeric import identity


class ReplayMemory():
    _actions_file = '/actions.npy'
    _frames_file = '/frames.npy'
    _rewards_file = '/rewards.npy'
    _terminal_file = '/terminal.npy'
    
    def __init__(self, size: int=1000000, input_shape: Tuple[int]=(84, 84), history_length: int=4, use_per=True) -> None:
        self.size = size
        self.input_shape = input_shape
        self.history_length = history_length
        self.use_per = use_per # from PER
        self.count = 0
        self.current = 0
        
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        self.terminal = np.empty(self.size, dtype=np.bool)
        self.priorities = np.zeros(self.size, dtype=np.float32) # from PER
        
    def add(self, action: int, reward: float, frame: np.ndarray, terminal: bool, clip_reward=True) -> None:
        assert frame.shape ==  self.input_shape, f'Shape of frame should be {self.input_shape} not {frame.shape}'
        
        if clip_reward:
            reward = np.sign(reward)
        
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal[self.current] = terminal
        self.priorities[self.current] = max(self.priorities.max(), 1)
        
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size
    
    def _get_valid_index(self) -> int:
        index: int = 0
        
        while True:
            index = random.randint(self.history_length, self.count - 1)
            if index < self.history_length:
                continue
            elif index >= self.current and index - self.history_length <= self.current:
                continue
            elif self.terminal[index - self.history_length:index].any():
                continue
            break
        
        return index
    
    def get_minibatch(self, batch_size: int=32, priority_scale: float=0.0) -> Tuple:
        assert self.count > self.history_length, 'Not enough data in Replay Memory to get minibatch'
        
        if self.use_per:
            scaled_priorities = self.priorities[self.history_length:self.count-1] ** priority_scale
            sample_probabilities = scaled_priorities / sum(scaled_priorities)
        
        indices: list = []
        for _ in range(batch_size):
            index = self._get_valid_index()
            indices.append(index)
            
        states = []
        new_states = []
        for i in indices:
            states.append(self.frames[i - self.history_length:i, ...])
            new_states.append(self.frames[i - self.history_length + 1:i+1, ...])
        
        # Why?
        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))
        
        if self.use_per:
            importance = (1 / self.count) * 1 / sample_probabilities[[index - self.history_length for index in indices]]
            importance = importance / importance.max()
            return states, self.actions[indices], self.rewards[indices], new_states, self.terminal[indices], importance, indices
        
        return states, self.actions[indices], self.rewards[indices], new_states, self.terminal[indices]
    
    def set_priorities(self, indices, errors, offset=0.1) ->  None:
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset
    
    def save(self, path) -> None:
        # for future update we will save the object parameters also
        if not os.path.isdir(path):
            os.mkdir(path)
        
        np.save(path + self._actions_file, self.actions)
        np.save(path + self._frames_file, self.frames)
        np.save(path + self._rewards_file, self.rewards)
        np.save(path + self._terminal_file, self.terminal)
    
    def load(self, path) -> None:
        self.actions = np.load(path + self._actions_file)
        self.frames = np.load(path + self._frames_file)
        self.rewards = np.load(path + self._rewards_file)
        self.terminal = np.load(path + self._terminal_file)