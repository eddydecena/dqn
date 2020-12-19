# Deep Q-Learning Implementation

Reinforcement Learning (RL) is a machine learning method that learns to solve tasks through trial and error. The combination of RL and Deep Learning is known as Deep Reinforcement Learning (DRL). One of the most important algorithms in the field of DRL is Deep Q-Networks (DQN), in this repo you have an implementation of this algorithm base on those papers. ([arXiv:1312.5602](https://arxiv.org/abs/1312.5602), [arXiv:1710.02298](https://arxiv.org/abs/1710.02298), [arXiv:1511.06581](https://arxiv.org/abs/1511.06581, https://arxiv.org/pdf/1511.05952.pdf))

This implementation is base on up to date technologies and good practice:

- python 3.8
- tensorflow 2.3
- Typing annotations
- Docker

# Documentation

```python
from dqn import DQNAgent
from dqn import Env

env = Env('name_env')
agent = DQNAgent(...)

# Training with all steps
num_frames = 0
while (num_frames < 30000000):
    env.reset(...)
    for _ in range(18000)
        action = agent.get_actions(...)
        env.step(action)
        agent.learn(...)
    
    num_frames = agent.count

# Evaluate with all steps
agent.eval(...)
```
