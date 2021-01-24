import time
from typing import Union
from typing import Optional
from typing import Tuple

from gym import Env
import numpy as np
import cv2 as cv
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from dqn.agents import DQNAgent
from dqn.environments import GameWrapper, game_wrapper

class Agent():
    def __init__(
        self, 
        env: Union[str, Env],
        input_shape: Tuple[int]=(84, 84),
        batch_size: int=4,
        history_length: int=4,
        learning_rate: float=0.00001,
        grad_descent_freq: int=4,
        target_update_freq: int=1000,
        max_frames: int=30000000,
        max_episode_length: int=18000,
        frames_between_eval: int=100000,
        eval_length: int=10000,
        gamma: float=0.99, # discount factor
        replay_buffer_start_size: int=50000,
        replay_buffer_size: int=100000,
        no_op_steps: int=20,
        save_path: Optional[str]='agent-checkpoints',
        load_replay_buffer: bool=True,
        use_tensorboard: bool=True,
        tensorboard_path: Optional[str]='trainer-tensorboard',
        priority_scale: float=0.7,
        clip_reward: bool=True,
        use_per: bool=False) -> None:
        
        self.max_frames = max_frames
        self.grad_descent_freq = grad_descent_freq
        self.target_update_freq = target_update_freq
        self.max_episode_length = max_episode_length
        self.frames_between_eval = frames_between_eval
        self.eval_length = eval_length
        self.gamma = gamma
        self.priority_scale = priority_scale
        self.clip_reward = clip_reward
        self.replay_buffer_start_size = replay_buffer_start_size
        self.batch_size = batch_size
        
        self.load_replay_buffer = load_replay_buffer
        self.use_tensorboard = use_tensorboard
        # self.tensorboard_path = tensorboard_path
        self.save_path = save_path
        
        self.game_wrapper = GameWrapper(env, no_op_steps=no_op_steps) 
        self.agent = DQNAgent(
                        n_actions=self.game_wrapper.env.action_space.n,
                        input_shape=input_shape,
                        batch_size=batch_size,
                        history_length=history_length,
                        learning_rate=learning_rate,
                        max_frames=max_frames,
                        replay_buffer_size=replay_buffer_size,
                        replay_buffer_start_size=replay_buffer_start_size,
                        use_per=use_per)
        
        self.logger = tf.summary.create_file_writer(tensorboard_path)
        
        self.frame_number: int = 0
        self.rewards: list = []
        self.loss_list: list = []
    
    def load(self, path: str) -> None:
        print(f'Loading from {path}')
        
        meta = self.agent.load(path, self.load_replay_buffer)
        
        self.frame_number = meta['frame_number']
        self.rewards = meta['rewards']
        self.loss_list = meta['loss_list']
        
        print(f'Loaded from {path}')
    
    def run(self) -> None:
        try:
            with self.logger.as_default():
                while self.frame_number < self.max_frames: # Why this need to be a while
                    eval_interval = 0
                    while eval_interval < self.frames_between_eval:
                        start_time = time.time()
                        self.game_wrapper.reset()
                        life_lost: bool = True
                        episode_reward_sum: int = 0
                        
                        for _ in range(self.max_episode_length):
                            action = self.agent.get_action(self.game_wrapper.state, frame_number=self.frame_number)
                            
                            processed_frame, reward, terminal, life_lost, frame = self.game_wrapper.step(action)
                            
                            # Show game
                            cv.imshow('Game', frame)
                            if cv.waitKey(1) & 0xFF == ord('q'):
                                raise KeyboardInterrupt
                            
                            self.frame_number += 1
                            eval_interval += 1
                            episode_reward_sum += reward
                            
                            self.agent.add_experience(
                                                action=action, 
                                                frame=processed_frame[:, :, 0], 
                                                reward=reward, 
                                                clip_reward=self.clip_reward, 
                                                terminal=terminal)
                            
                            if self.frame_number % self.grad_descent_freq == 0 and self.frame_number > self.replay_buffer_start_size:
                                loss, _ = self.agent.learn(
                                                gamma=self.gamma, 
                                                frame_number=self.frame_number, 
                                                priority_scale=self.priority_scale)
                                self.loss_list.append(loss)
                            
                            if self.frame_number % self.target_update_freq == 0 and self.frame_number > self.replay_buffer_start_size:
                                self.agent.update_target_network()
                            
                            if terminal:
                                terminal = False
                                break
                        self.rewards.append(episode_reward_sum)
                        
                        if len(self.rewards) % 10 == 0:
                            if self.use_tensorboard:
                                tf.summary.scalar('Reward', np.mean(self.rewards[-10:]), self.frame_number)
                                tf.summary.scalar('Loss', np.mean(self.loss_list[-100:]), self.frame_number)
                                self.logger.flush()
                            
                            print(f'Game number: {str(len(self.rewards)).zfill(6)}  Frame number: {str(self.frame_number).zfill(8)}  Average reward: {np.mean(self.rewards[-10:]):0.1f}  Time taken: {(time.time() - start_time):.1f}s')
                    
                    terminal = True
                    eval_rewards = []
                    eval_frame_number = 0 # may a should remove this
                    
                    for _ in range(self.eval_length):
                        if terminal:
                            game_wrapper.reset(evaluation=True)
                            life_lost = True
                            episode_reward_sum = 0
                            terminal = False
                        
                        action = 1 if life_lost else self.agent.get_action(self.frame_number, game_wrapper.state, evaluation=True)
                        
                        _, reward, terminal, life_lost = game_wrapper.step(action)
                        eval_frame_number = 0
                        episode_reward_sum += 1
                        
                        if terminal:
                            eval_rewards.append(episode_reward_sum)
                    
                    if len(eval_rewards) > 0:
                        final_score = np.mean(eval_rewards)
                    else:
                        final_score = episode_reward_sum
                    
                    print('Evaluation score: {final_score}')
                    if self.use_tensorboard:
                        tf.summary.scalar('Evaluation score', final_score, self.frame_number)
                        writer.flush()
                    
                    if len(self.rewards) > 300 and self.save_path is not None:
                        agent.save(f'{self.save_path}/save-{str(frame_number).zfill(8)}', frame_number=self.frame_number, rewards=self.rewards, loss_list=self.loss_list)
        except KeyboardInterrupt:
            self._save_on_interrupt()
    
    def _save_on_interrupt(self):
        cv.destroyAllWindows()
        
        print('\nTraining exited early.')
        self.logger.close()
        
        if self.save_path is None:
            try:
                self.save_path = input('Would you like to save the trained modelk? If so, type in a path, otherwise, interrupt with ctrl + c')
            except KeyboardInterrupt:
                print('\nClosing...')
        
        if self.save_path is not None:
            print('Saving...')
            self.agent.save(
                        f'{self.save_path}/save-{str(self.frame_number).zfill(8)}', 
                        frame_number=self.frame_number, 
                        rewards=self.rewards, 
                        loss_list=self.loss_list)
            print('Saved.')