import random
from typing import Union
from typing import Optional

import gym
import numpy as np
from dqn.preprocessing import frame_processor

class GameWrapper():
    def __init__(self, env: Union[str, gym.Env], no_op_steps: int=10, history_length: int=4) -> None:
        self.env = gym.make(env) if type(env) == str else env
        self.no_op_steps = no_op_steps
        self.history_length = history_length
        
        self.state = None
        self.last_lives = 0
    
    def reset(self, evaluation=False):
        self.frame = self.env.reset()
        self.last_lives = 0
        
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.env.step(1)
        
        self.state = np.repeat(frame_processor(self.frame), self.history_length, axis=2)
    
    def step(self, action: int, render_mode: Optional[str]='rgb_array'):
        next_frame, reward, terminal, info = self.env.step(action)
        
        if info['ale.lives'] < self.last_lives:
            life_lost = True
        else:
            life_lost = terminal
        self.last_lives = info['ale.lives']
        
        processed_frame = frame_processor(next_frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2)
        
        if render_mode == 'rgb_array':
            return processed_frame, reward, terminal, life_lost, self.env.render(render_mode)
        elif render_mode == 'human':
            self.env.render()