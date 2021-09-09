#!/usr/bin/env python
# coding: utf-8

# https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e

# # gym interface

# Each environment must implement the following *gym* interface:

# 
# 
# ```python
# import gym
# from gym import spaces
# 
# class CustomEnv(gym.Env):
#   """Custom Environment that follows gym interface"""
#   metadata = {'render.modes': ['human']}
# 
#   def __init__(self, arg1, arg2, ...):
#     super(CustomEnv, self).__init__()    # Define action and observation space
#     # They must be gym.spaces objects    # Example when using discrete actions:
#     self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)    # Example for using image as input:
#     self.observation_space = spaces.Box(low=0, high=255, shape=
#                     (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
# 
#   def step(self, action):
#     # Execute one time step within the environment
#       
#     
#   def reset(self):
#     # Reset the state of the environment to an initial state
#       
#     
#   def render(self, mode='human', close=False):
#     # Render the environment to the screen
#     
# ```

# # house made goldmine - discrete mode

# ![image.png](attachment:image.png)

# ## env

# In[40]:


import gym
from gym import spaces
import random

class GoldMine(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(GoldMine, self).__init__()
        #actions: move on the grid, by 1 in any direction: N, S, W, E. Or stay at same position
        #0 - N, 1 - S, 2 - W, 3 - E, 4 - no move
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0.0, high=5.0, shape=(2,))
    
    def reset(self):
        self.current_position = self.observation_space.sample()
        self.global_reward = 0
        print(f'reset at position {self.current_position}')
        return self.current_position
        
    def step(self, action):
        if (action == 0) & (self.current_position[1] < 4):
            self.current_position[1] += 1
        if (action == 1) & (self.current_position[1] > 1):
            self.current_position[1] -= 1
        if (action == 2) & (self.current_position[0] > 1):
            self.current_position[0] -= 1
        if (action == 3) & (self.current_position[0] < 4):
            self.current_position[0] += 1
        reward = -1
        done = False
        if (self.current_position[0] > 4) & (self.current_position[1] > 4):
            reward = 10
            done = True
        self.global_reward += reward
        return self.current_position, reward, done, {}
            
    
    def render(self):
        print(f'position {self.current_position}, done {done}, global_reward {self.global_reward}')
            
        


# In[41]:


env = GoldMine()
env.reset()
for i in range(100):
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    env.render()
    if done: env.reset()

    


# ## checkenv

# In[42]:


import stable_baselines3
from stable_baselines3.common.env_checker import check_env

check_env(env)


# ## train DQN

# In[46]:


from stable_baselines3 import DQN

del model
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")

model.learn(total_timesteps=100000, log_interval=4, tb_log_name="goldmine simple reward")


# In[ ]:


get_ipython().system('tensorboard --logdir ./tensorboard/')


# ![image.png](attachment:image.png)

# In[28]:


model.save('dqn_simplest_model')


# ## run optimization

# In[44]:


env.reset()
for i in range(100):
    action, _ = model.predict(env.current_position)
    print(f'action {action}')
    obs, rewards, done, info = env.step(action)
    env.render()
    if done: break


# # goldmine with different reward (incentive to success)

# In[1]:


import gym
from gym import spaces
import random
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN


class GoldMine2(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(GoldMine2, self).__init__()
        #actions: move on the grid, by 1 in any direction: N, S, W, E. Or stay at same position
        #0 - N, 1 - S, 2 - W, 3 - E, 4 - no move
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0.0, high=5.0, shape=(2,))
    
    def reset(self):
        self.current_position = self.observation_space.sample()
        self.global_reward = 0
        print(f'reset at position {self.current_position}')
        return self.current_position
        
    def step(self, action):
        if (action == 0) & (self.current_position[1] < 4):
            self.current_position[1] += 1
        if (action == 1) & (self.current_position[1] > 1):
            self.current_position[1] -= 1
        if (action == 2) & (self.current_position[0] > 1):
            self.current_position[0] -= 1
        if (action == 3) & (self.current_position[0] < 4):
            self.current_position[0] += 1
        reward = -0.001
        done = False
        if (self.current_position[0] > 4) & (self.current_position[1] > 4):
            reward = 10
            done = True
        self.global_reward += reward
        return self.current_position, reward, done, {}
            
    
    def render(self):
        print(f'position {self.current_position}, done {done}, global_reward {self.global_reward}')
        
        


env_gold2 = GoldMine2()
check_env(env_gold2)

model_gold2 = DQN("MlpPolicy", env_gold2, verbose=1, tensorboard_log="./tensorboard/")
model_gold2.learn(total_timesteps=100000, log_interval=4, tb_log_name="goldmine2 smooth reward")

env_gold2.reset()
for i in range(100):
    action, _ = model_gold2.predict(env_gold2.current_position)
    print(f'action {action}')
    obs, rewards, done, info = env_gold2.step(action)
    env_gold2.render()
    if done: break


# # goldmine with cheating reward based on distance

# In[2]:


import gym
from gym import spaces
import random
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from math import sqrt

class GoldMine3(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(GoldMine3, self).__init__()
        #actions: move on the grid, by 1 in any direction: N, S, W, E. Or stay at same position
        #0 - N, 1 - S, 2 - W, 3 - E, 4 - no move
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0.0, high=5.0, shape=(2,))
    
    def reset(self):
        self.current_position = self.observation_space.sample()
        self.global_reward = 0
        print(f'reset at position {self.current_position}')
        return self.current_position
        
    def step(self, action):
        if (action == 0) & (self.current_position[1] < 4):
            self.current_position[1] += 1
        if (action == 1) & (self.current_position[1] > 1):
            self.current_position[1] -= 1
        if (action == 2) & (self.current_position[0] > 1):
            self.current_position[0] -= 1
        if (action == 3) & (self.current_position[0] < 4):
            self.current_position[0] += 1
        #we will calculate a distance to the gold pot
        reward = -sqrt((5-self.current_position[0])**2+(5-self.current_position[1])**2)
        done = False
        if (self.current_position[0] > 4) & (self.current_position[1] > 4):
            reward = 100
            done = True
        self.global_reward += reward
        return self.current_position, reward, done, {}
            
    
    def render(self):
        print(f'position {self.current_position}, done {done}, global_reward {self.global_reward}')
        
        


env_gold3 = GoldMine3()
check_env(env_gold3)

model_gold3 = DQN("MlpPolicy", env_gold3, verbose=1, tensorboard_log="./tensorboard/")
model_gold3.learn(total_timesteps=100000, log_interval=4, tb_log_name="goldmine3 cheating reward")

env_gold3.reset()
for i in range(100):
    action, _ = model_gold3.predict(env_gold3.current_position)
    print(f'action {action}')
    obs, rewards, done, info = env_gold3.step(action)
    env_gold3.render()
    if done: break


# # goldmine with action box and DDPG

# In[3]:


import gym
from gym import spaces
import random
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, DDPG
from math import sqrt

class GoldMine4(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(GoldMine4, self).__init__()
        #actions: move on the grid, by continuous value in -2,2 in any direction: N, S, W, E. Or stay at same position
        #action[0] - N (positive) S (negative)
        #action[1] - E (positive) W (negative)
        #0,0 no move
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(2,))
        self.observation_space = spaces.Box(low=0.0, high=5.0, shape=(2,))
    
    def reset(self):
        self.current_position = self.observation_space.sample()
        self.global_reward = 0
        print(f'reset at position {self.current_position}')
        return self.current_position
        
    def step(self, action):
        if ((action[0]+self.current_position[1]<5) & (action[0]+self.current_position[1]>0)):
            self.current_position[1] += action[0]
        if ((action[1]+self.current_position[0]<5) & (action[1]+self.current_position[0]>0)):
            self.current_position[0] += action[1]

        #we will calculate a distance to the gold pot
        reward = -10
        done = False
        if (self.current_position[0] > 4) & (self.current_position[1] > 4):
            reward = 100
            done = True
        self.global_reward += reward
        return self.current_position, reward, done, {}
            
    
    def render(self):
        print(f'position {self.current_position}, done {done}, global_reward {self.global_reward}')
        
        


env_gold4 = GoldMine4()
check_env(env_gold4)

model_gold4 = DDPG("MlpPolicy", env_gold4, verbose=1, tensorboard_log="./tensorboard/")
model_gold4.learn(total_timesteps=100000, log_interval=4, tb_log_name="goldmine4 ddpg")

env_gold4.reset()
for i in range(100):
    action, _ = model_gold4.predict(env_gold4.current_position)
    print(f'action {action}')
    obs, rewards, done, info = env_gold4.step(action)
    env_gold4.render()
    if done: break


# In[ ]:




