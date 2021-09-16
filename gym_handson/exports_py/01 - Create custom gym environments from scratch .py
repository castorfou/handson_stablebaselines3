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

# In[1]:


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
            
        


# In[21]:


env = GoldMine()
env.reset()
for i in range(100):
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    env.render()
    if done: env.reset()

    


# ## checkenv

# In[25]:


import stable_baselines3
from stable_baselines3.common.env_checker import check_env

check_env(env)


# ## train DQN

# In[26]:


from stable_baselines3 import DQN

# del model
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")

model.learn(total_timesteps=100000, log_interval=4, tb_log_name="goldmine simple reward")


# ## tensorboard

# In[ ]:


get_ipython().system('tensorboard --logdir ./tensorboard/')


# ![image.png](attachment:image.png)

# ## save model

# In[28]:


model.save('dqn_simplest_model')


# ## run optimization

# In[27]:


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


# # goldmine with action box (continuous)

# In[28]:


import gym
from gym import spaces
import random
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, DDPG
from math import sqrt
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

class GoldMine4(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(GoldMine4, self).__init__()
        #actions: move on the grid, by continuous value in -2,2 in any direction: N, S, W, E. Or stay at same position
        #action[0] - N (positive) S (negative)
        #action[1] - E (positive) W (negative)
        #0,0 no move
        #"We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) ", we will multiply effect by 2
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.observation_space = spaces.Box(low=0.0, high=5.0, shape=(2,))
        
    
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
        
        


# In[29]:


env_gold4 = GoldMine4()
env_gold4.reset()
for i in range(100):
    action = env_gold4.action_space.sample()
    obs, rewards, done, info = env_gold4.step(action)
    env_gold4.render()
    if done: env_gold4.reset()


# ## use of DDPG
# 
# https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html

# In[30]:


env_gold4 = GoldMine4()
check_env(env_gold4)

# The noise objects for DDPG
n_actions = env_gold4.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


model_gold4 = DDPG("MlpPolicy", env_gold4, action_noise=action_noise, verbose=2,tensorboard_log="./tensorboard/")
model_gold4.learn(total_timesteps=10000, log_interval=4, tb_log_name="goldmine4 ddpg")

env_gold4.reset()
for i in range(100):
    action, _ = model_gold4.predict(env_gold4.current_position)
    print(f'action {action}')
    obs, rewards, done, info = env_gold4.step(action)
    env_gold4.render()
    if done: break
env_gold4.close()


# ## use of A2C
# 
# https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html

# In[44]:


from stable_baselines3 import A2C

env_gold4 = GoldMine4()
check_env(env_gold4)

model_gold_a2c = A2C("MlpPolicy", env_gold4, verbose=2,tensorboard_log="./tensorboard/")
model_gold_a2c.learn(total_timesteps=10000, log_interval=4, tb_log_name="goldmine4 a2c")

env_gold4.reset()
for i in range(100):
    action, _ = model_gold_a2c.predict(env_gold4.current_position)
    print(f'action {action}')
    obs, rewards, done, info = env_gold4.step(action)
    env_gold4.render()
    if done: break
env_gold4.close()


# ## use of PPO
# 
# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

# In[45]:


from stable_baselines3 import PPO

env_gold4 = GoldMine4()
check_env(env_gold4)

model_gold_ppo = PPO("MlpPolicy", env_gold4, verbose=2,tensorboard_log="./tensorboard/")
model_gold_ppo.learn(total_timesteps=10000, log_interval=4, tb_log_name="goldmine4 ppo")

env_gold4.reset()
for i in range(100):
    action, _ = model_gold_ppo.predict(env_gold4.current_position)
    print(f'action {action}')
    obs, rewards, done, info = env_gold4.step(action)
    env_gold4.render()
    if done: break
env_gold4.close()


# ## use of SAC
# 
# https://stable-baselines3.readthedocs.io/en/master/modules/sac.html

# In[46]:


from stable_baselines3 import SAC

env_gold4 = GoldMine4()
check_env(env_gold4)

model_gold_sac = SAC("MlpPolicy", env_gold4, verbose=2,tensorboard_log="./tensorboard/")
model_gold_sac.learn(total_timesteps=10000, log_interval=4, tb_log_name="goldmine4 sac")

env_gold4.reset()
for i in range(100):
    action, _ = model_gold_sac.predict(env_gold4.current_position)
    print(f'action {action}')
    obs, rewards, done, info = env_gold4.step(action)
    env_gold4.render()
    if done: break
env_gold4.close()


# ## use of TD3
# 
# https://stable-baselines3.readthedocs.io/en/master/modules/td3.html

# In[47]:


from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env_gold4 = GoldMine4()
check_env(env_gold4)

n_actions = env_gold4.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


model_gold_td3 = TD3("MlpPolicy", env_gold4, action_noise=action_noise, verbose=2,tensorboard_log="./tensorboard/")
model_gold_td3.learn(total_timesteps=10000, log_interval=4, tb_log_name="goldmine4 td3")

env_gold4.reset()
for i in range(100):
    action, _ = model_gold_td3.predict(env_gold4.current_position)
    print(f'action {action}')
    obs, rewards, done, info = env_gold4.step(action)
    env_gold4.render()
    if done: break
env_gold4.close()


# In[ ]:




