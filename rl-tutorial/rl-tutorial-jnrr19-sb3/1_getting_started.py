#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/1_getting_started.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Stable Baselines3 Tutorial - Getting Started
# 
# Github repo: https://github.com/araffin/rl-tutorial-jnrr19/tree/sb3/
# 
# Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
# 
# Documentation: https://stable-baselines3.readthedocs.io/en/master/
# 
# RL Baselines3 zoo: https://github.com/DLR-RM/rl-baselines3-zoo
# 
# [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) is a collection of pre-trained Reinforcement Learning agents using Stable-Baselines3.
# 
# It also provides basic scripts for training, evaluating agents, tuning hyperparameters and recording videos.
# 
# 
# ## Introduction
# 
# In this notebook, you will learn the basics for using stable baselines library: how to create a RL model, train it and evaluate it. Because all algorithms share the same interface, we will see how simple it is to switch from one algorithm to another.
# 
# 
# ## Install Dependencies and Stable Baselines3 Using Pip
# 
# List of full dependencies can be found in the [README](https://github.com/DLR-RM/stable-baselines3).
# 
# 
# ```
# pip install stable-baselines3[extra]
# ```

# In[1]:


get_ipython().system('apt-get install ffmpeg freeglut3-dev xvfb  # For visualization')
get_ipython().system('pip install stable-baselines3[extra]')


# ## Imports

# Stable-Baselines3 works on environments that follow the [gym interface](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html).
# You can find a list of available environment [here](https://gym.openai.com/envs/#classic_control).
# 
# It is also recommended to check the [source code](https://github.com/openai/gym) to learn more about the observation and action space of each env, as gym does not have a proper documentation.
# Not all algorithms can work with all action spaces, you can find more in this [recap table](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html)

# In[1]:


import gym
import numpy as np


# The first thing you need to import is the RL model, check the documentation to know what you can use on which problem

# In[2]:


from stable_baselines3 import PPO


# The next thing you need to import is the policy class that will be used to create the networks (for the policy/value functions).
# This step is optional as you can directly use strings in the constructor: 
# 
# ```PPO('MlpPolicy', env)``` instead of ```PPO(MlpPolicy, env)```
# 
# Note that some algorithms like `SAC` have their own `MlpPolicy`, that's why using string for the policy is the recommened option.

# In[3]:


from stable_baselines3.ppo.policies import MlpPolicy


# ## Create the Gym env and instantiate the agent
# 
# For this example, we will use CartPole environment, a classic control problem.
# 
# "A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. "
# 
# Cartpole environment: [https://gym.openai.com/envs/CartPole-v1/](https://gym.openai.com/envs/CartPole-v1/)
# 
# ![Cartpole](https://cdn-images-1.medium.com/max/1143/1*h4WTQNVIsvMXJTCpXm_TAw.gif)
# 
# 
# We chose the MlpPolicy because the observation of the CartPole task is a feature vector, not images.
# 
# The type of action to use (discrete/continuous) will be automatically deduced from the environment action space
# 
# Here we are using the [Proximal Policy Optimization](https://stable-baselines3.readthedocs.io/en/master/modules/ppo2.html) algorithm, which is an Actor-Critic method: it uses a value function to improve the policy gradient descent (by reducing the variance).
# 
# It combines ideas from [A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html) (having multiple workers and using an entropy bonus for exploration) and [TRPO](https://stable-baselines.readthedocs.io/en/master/modules/trpo.html) (it uses a trust region to improve stability and avoid catastrophic drops in performance).
# 
# PPO is an on-policy algorithm, which means that the trajectories used to update the networks must be collected using the latest policy.
# It is usually less sample efficient than off-policy alorithms like [DQN](https://stable-baselines.readthedocs.io/en/master/modules/dqn.html), [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) or [TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html), but is much faster regarding wall-clock time.
# 

# In[4]:


env = gym.make('CartPole-v1')

model = PPO(MlpPolicy, env, verbose=0)


# We create a helper function to evaluate the agent:

# In[5]:


def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


# Let's evaluate the un-trained agent, this should be a random agent.

# In[7]:


# Random Agent, before training
mean_reward_before_train = evaluate(model, num_episodes=100)


# Stable-Baselines already provides you with that helper:

# In[8]:


from stable_baselines3.common.evaluation import evaluate_policy


# In[9]:


mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


# ## Train the agent and evaluate it

# In[ ]:


# Train the agent for 10000 steps
model.learn(total_timesteps=10000)


# In[ ]:


# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


# Apparently the training went well, the mean reward increased a lot ! 

# ### Prepare video recording

# In[ ]:


# Set up fake display; otherwise rendering will fail
import os
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'


# In[ ]:


import base64
from pathlib import Path

from IPython import display as ipythondisplay

def show_videos(video_path='', prefix=''):
  """
  Taken from https://github.com/eleurent/highway-env

  :param video_path: (str) Path to the folder containing videos
  :param prefix: (str) Filter the video, showing only the only starting with this prefix
  """
  html = []
  for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
      video_b64 = base64.b64encode(mp4.read_bytes())
      html.append('''<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>'''.format(mp4, video_b64.decode('ascii')))
  ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


# We will record a video using the [VecVideoRecorder](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecvideorecorder) wrapper, you will learn about those wrapper in the next notebook.

# In[ ]:


from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
  eval_env = DummyVecEnv([lambda: gym.make(env_id)])
  # Start the video at step=0 and record 500 steps
  eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix=prefix)

  obs = eval_env.reset()
  for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, _, _ = eval_env.step(action)

  # Close the video recorder
  eval_env.close()


# ### Visualize trained agent
# 
# 

# In[17]:


record_video('CartPole-v1', model, video_length=500, prefix='ppo2-cartpole')


# In[ ]:


show_videos('videos', prefix='ppo2')


# ## Bonus: Train a RL Model in One Line
# 
# The policy class to use will be inferred and the environment will be automatically created. This works because both are [registered](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html).

# In[ ]:


model = PPO('MlpPolicy', "CartPole-v1", verbose=1).learn(1000)


# ## Conclusion
# 
# In this notebook we have seen:
# - how to define and train a RL model using stable baselines3, it takes only one line of code ;)

# In[ ]:




