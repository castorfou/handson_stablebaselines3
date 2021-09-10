#!/usr/bin/env python
# coding: utf-8

# # Janus gym environment

# In[53]:


import gym
from gym import spaces
import random
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, DDPG
from math import sqrt
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

class Janus(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(Janus, self).__init__()
        #actions: move on the grid, by continuous value in -1,1 
        #0,0 no move
        #based on 98 controlable parameters
        #"We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) ", we will multiply effect by 2
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(98,))
        
        # all the observation_space
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(325,))
        
    
    def reset(self):
        self.current_position = self.observation_space.sample()
        self.last_action = np.array([])
        self.last_effect = False
        self.global_reward = 0
        print(f'reset at position {self.current_position}')
        return self.current_position
        
    def step(self, action):
        action *= 2
        effect = False
        if ((action[0]+self.current_position[1]<5) & (action[0]+self.current_position[1]>0)):
            effect = True
            self.current_position[1] += action[0]
        if ((action[1]+self.current_position[0]<5) & (action[1]+self.current_position[0]>0)):
            effect = True
            self.current_position[0] += action[1]

        reward = -10
        
        if (self.last_action.size > 0):
            ## can be used to compare to last action if it is valid
            pass
        self.last_action = action
        self.last_effect = effect

        done = False
        if (self.current_position[0] > 4) & (self.current_position[1] > 4):
            reward = 100
            done = True
        self.global_reward += reward
        return self.current_position, reward, done, {}
            
    
    def render(self):
        print(f'position {self.current_position}, action {self.last_action}, effect {self.last_effect}, done {done}, global_reward {self.global_reward}')
        
    def convert_to_real_obs(self, observation, observation_dataset):
        '''
        observation: instance of observation_space
        '''
        return (observation +np.ones(self.observation_space.shape))/2 * (observation_dataset.max()-observation_dataset.min())+observation_dataset.min()
        


# # load of data

# In[7]:


import numpy as np
import pandas as pd

template_filename = 'data/dataset-S_public/public/dataset_S-{}.csv'


file1 = pd.read_csv(template_filename.format('file1'), index_col=0)
file3 = pd.read_csv(template_filename.format('file3'), index_col=0)


# In[33]:


print(f'valeur min dans le dataset {min(file3.min())}, \ntop 5 min \n {file3.min()[file3.min().argsort()].head(5)}')
print(f'valeur maxi dans le dataset {max(file3.max())}, \ntop 5 max \n {file3.max()[file3.max().argsort(reversed)].tail(5)}')


# In[34]:


file3.describe()


# In[54]:


env_test = Janus()
from stable_baselines3.common.env_checker import check_env

# check_env(env_test)

env_test.convert_to_real_obs(env_test.observation_space.sample(), file3)


# In[63]:


obs = env_test.observation_space.sample()
arr = env_test.convert_to_real_obs(obs, file3)

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Observation in real world(top), observation space (bottom)')
ax1.plot(np.arange(0, len(arr)), arr)
ax2.plot(np.arange(0, len(arr)), obs)
plt.show()


# # prediction and reward calculation

# In[ ]:




