#!/usr/bin/env python
# coding: utf-8

# # Stable Baselines3 Hands-on Session - RLVS
# 
# Github repo: https://github.com/araffin/rl-handson-rlvs21
# 
# Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
# 
# Documentation: https://stable-baselines3.readthedocs.io/en/master/
# 
# SB3 Contrib: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
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
# In this notebook, you will learn the basics for using stable baselines3 library: how to create a RL model, train it and evaluate it. Because all algorithms share the same interface, we will see how simple it is to switch from one algorithm to another.
# You will also learn how to define a gym wrapper and callback to customise the training.
# We will finish this session by trying out multiprocessing and have a hyperparameter tuning challenge.
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

# In[ ]:


get_ipython().system('apt-get install ffmpeg freeglut3-dev xvfb  # For visualization')


# In[ ]:


get_ipython().system('pip install stable-baselines3[extra]')


# In[2]:


# Optional: install SB3 contrib to have access to additional algorithms
get_ipython().system('pip install sb3-contrib')


# # Part I: Getting Started

# ## First steps with the Gym interface
# 
# An environment that follows the [gym interface](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html) is quite simple to use.
# It provides to this user mainly three methods:
# - `reset()` called at the beginning of an episode, it returns an observation
# - `step(action)` called to take an action with the environment, it returns the next observation, the immediate reward, whether the episode is over and additional information
# - (Optional) `render(method='human')` which allow to visualize the agent in action. Note that graphical interface does not work on google colab, so we cannot use it directly (we have to rely on `method='rbg_array'` to retrieve an image of the scene
# 
# Under the hood, it also contains two useful properties:
# - `observation_space` which one of the gym spaces (`Discrete`, `Box`, ...) and describe the type and shape of the observation
# - `action_space` which is also a gym space object that describes the action space, so the type of action that can be taken
# 
# The best way to learn about gym spaces is to look at the [source code](https://github.com/openai/gym/tree/master/gym/spaces), but you need to know at least the main ones:
# - `gym.spaces.Box`: A (possibly unbounded) box in $R^n$. Specifically, a Box represents the Cartesian product of n closed intervals. Each interval has the form of one of [a, b], (-oo, b], [a, oo), or (-oo, oo). Example: A 1D-Vector or an image observation can be described with the Box space.
# ```python
# # Example for using image as input:
# observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
# ```                                       
# 
# - `gym.spaces.Discrete`: A discrete space in $\{ 0, 1, \dots, n-1 \}$
#   Example: if you have two actions ("left" and "right") you can represent your action space using `Discrete(2)`, the first action will be 0 and the second 1.
# 
# 
# 
# [Documentation on custom env](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)
# 
# Below you can find an example of a custom environment:

# In[3]:


from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common.env_checker import check_env

GymObs = Union[Tuple, Dict, np.ndarray, int]

class CustomEnv(gym.Env):
  """
  Minimal custom environment to demonstrate the Gym interface.
  """
  def __init__(self):
    super(CustomEnv, self).__init__()
    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,))
    self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))

  def reset(self) -> GymObs:
    """
    Called at the beginning of an episode.
    :return: the first observation of the episode
    """
    return self.observation_space.sample()

  def step(self, action: Union[int, np.ndarray]) -> Tuple[GymObs, float, bool, Dict]:
    """
    Step into the environment.
    :return: A tuple containing the new observation, the reward signal, 
      whether the episode is over and additional informations.
    """
    obs = self.observation_space.sample()
    reward = 1.0
    done = False
    info = {}
    return obs, reward, done, info

env = CustomEnv()
# Check your custom environment
# this will print warnings and throw errors if needed
check_env(env)


# ## Imports

# Stable-Baselines3 works on environments that follow the [gym interface](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html).
# You can find a list of available environment [here](https://gym.openai.com/envs/#classic_control).
# 
# It is also recommended to check the [source code](https://github.com/openai/gym) to learn more about the observation and action space of each env, as gym does not have a proper documentation.
# Not all algorithms can work with all action spaces, you can find more in this [recap table](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html)

# In[4]:


import gym
import numpy as np


# The first thing you need to import is the RL model, check the documentation to know what you can use on which problem

# In[5]:


from stable_baselines3 import PPO, A2C, SAC, TD3, DQN


# In[6]:


# Algorithms from the contrib repo
# https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
from sb3_contrib import QRDQN, TQC


# The next thing you need to import is the policy class that will be used to create the networks (for the policy/value functions).
# This step is optional as you can directly use strings in the constructor: 
# 
# ```PPO("MlpPolicy", env)``` instead of ```PPO(MlpPolicy, env)```
# 
# Note that some algorithms like `SAC` have their own `MlpPolicy`, that's why using string for the policy is the recommended option.

# In[7]:


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
# Here we are using the [Proximal Policy Optimization](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) algorithm, which is an Actor-Critic method: it uses a value function to improve the policy gradient descent (by reducing the variance).
# 
# It combines ideas from [A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html) (having multiple workers and using an entropy bonus for exploration) and [TRPO](https://stable-baselines.readthedocs.io/en/master/modules/trpo.html) (it uses a trust region to improve stability and avoid catastrophic drops in performance).
# 
# PPO is an on-policy algorithm, which means that the trajectories used to update the networks must be collected using the latest policy.
# It is usually less sample efficient than off-policy alorithms like [DQN](https://stable-baselines.readthedocs.io/en/master/modules/dqn.html), [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) or [TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html), but is much faster regarding wall-clock time.
# 

# In[8]:


# Create the gym Env
env = gym.make('CartPole-v1')
# Create the RL agent
model = PPO('MlpPolicy', env, verbose = 1)


# ### Using the model to predict actions

# In[9]:


print(env.observation_space)
print(env.action_space)


# In[10]:


# Retrieve first observation
obs = env.reset()


# In[11]:


# Predict the action to take given the observation
action, _ = model.predict(obs, deterministic=True)


# In[12]:


# We are using discrete actions, therefore `action` is an int
assert env.action_space.contains(action)

print(action)


# Step in the environment

# In[13]:


obs, reward, done, infos = env.step(action)


# In[14]:


print(f"obs_shape={obs.shape}, reward={reward}, done? {done}")


# In[16]:


# Reset the env at the end of an episode
if done:
  obs = env.reset()


# ### Exercise (10 minutes): write the function to evaluate the agent
# 
# This function will be used to evaluate the performance of an RL agent.
# Thanks to Stable Baselines3 interface, it will work with any SB3 algorithms and any Gym environment.
# 
# See docstring of the function for what is expected as input/output.

# In[22]:


from stable_baselines3.common.base_class import BaseAlgorithm


def evaluate(
    model: BaseAlgorithm,
    env: gym.Env,
    n_eval_episodes: int = 100,
    deterministic: bool = False,
) -> float:
    """
    Evaluate an RL agent for `n_eval_episodes`.

    :param model: the RL Agent
    :param env: the gym Environment
    :param n_eval_episodes: number of episodes to evaluate it
    :param deterministic: Whether to use deterministic or stochastic actions
    :return: Mean reward for the last `n_eval_episodes`
    """
    ### YOUR CODE HERE
    # TODO: run `n_eval_episodes` episodes in the Gym env
    # using the RL agent and keep track of the total reward
    # collected for each episode.
    # Finally, compute the mean and print it
    rewards_list=[]
    for i in range(n_eval_episodes):
      obs = env.reset()
      done=False
      reward_sum=0
      while(not done):
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, rewards, done, info = env.step(action)
        reward_sum+=rewards
      rewards_list.append(reward_sum)
    mean_episode_reward = np.sum(rewards_list)/n_eval_episodes
    print(f"mean_reward={mean_episode_reward}, number_episodes={n_eval_episodes}")

    ### END OF YOUR CODE
    return mean_episode_reward


# Let's evaluate the un-trained agent, this should be a random agent.

# In[23]:


env = gym.make('CartPole-v1')
model = PPO('MlpPolicy',  env, seed=1,verbose=1)


# In[24]:


# Random Agent, before training
mean_reward_before_train = evaluate(model, env, n_eval_episodes=100, deterministic=False)


# Stable-Baselines already provides you with that helper (the actual implementation is a little more advanced):

# In[25]:


from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


# In[26]:


# The Monitor wrapper allows to keep track of the training reward and other infos (useful for plotting)
env = Monitor(env)


# In[27]:


# Seed to compare to previous implementation
env.seed(42)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


# ## Train the agent and evaluate it

# In[28]:


# Train the agent for 10000 steps
model.learn(total_timesteps=10000)


# In[29]:


# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


# Apparently the training went well, the mean reward increased a lot! 

# ### Prepare video recording

# In[30]:


# Set up fake display; otherwise rendering will fail
import os
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'


# In[31]:


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


# We will record a video using the [VecVideoRecorder](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecvideorecorder) wrapper, you can learn more about those wrappers in our Documentation.

# In[32]:


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
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = eval_env.step(action)

  # Close the video recorder
  eval_env.close()


# ### Visualize trained agent
# 
# 

# In[33]:


record_video('CartPole-v1', model, video_length=500, prefix='ppo-cartpole')


# In[34]:


show_videos('videos', prefix='ppo')


# ### Exercise (5 minutes): Save, Load The Model and that the loading was correct
# 
# Save the model and then load it.
# 
# Don't forget to check that loading went well: the model must predict the same actions given the same  observations.

# In[35]:


# Sample observations using the environment observation space
observations = np.array([env.observation_space.sample() for _ in range(10)])
# Predict actions on those observations using trained model


action_before_saving, _ = model.predict(observations, deterministic=True)


# In[36]:


# Save the model
model.save("ppo_cartpole")


# In[37]:


# Delete the model (to demonstrate loading)
del model


# In[38]:


get_ipython().system('ls *.zip')


# In[39]:


# Load the model
model = PPO.load('ppo_cartpole')


# In[40]:


# Predict actions on the observations with the loaded model
action_after_loading, _ = model.predict(observations, deterministic=True)


# In[41]:


# Check that the predictions are the same
assert np.allclose(action_before_saving, action_after_loading), "Somethng went wrong in the loading"


# ## Bonus: Train a RL Model in One Line
# 
# The policy class to use will be inferred and the environment will be automatically created. This works because both are [registered](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html).

# In[ ]:


model = PPO('MlpPolicy', "CartPole-v1", verbose=1).learn(1000)


# # Part II: Gym Wrappers
# 
# 
# In this part, you will learn how to use *Gym Wrappers* which allow to do monitoring, normalization, limit the number of steps, feature augmentation, ...
# 

# ## Anatomy of a gym wrapper

# A gym wrapper follows the [gym](https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html) interface: it has a `reset()` and `step()` method.
# 
# Because a wrapper is *around* an environment, we can access it with `self.env`, this allow to easily interact with it without modifying the original env.
# There are many wrappers that have been predefined, for a complete list refer to [gym documentation](https://github.com/openai/gym/tree/master/gym/wrappers)

# In[42]:


class CustomWrapper(gym.Wrapper):
  """
  :param env:  Gym environment that will be wrapped
  """
  def __init__(self, env: gym.Env):
    # Call the parent constructor, so we can access self.env later
    super().__init__(env)
  
  def reset(self):
    """
    Reset the environment 
    """
    obs = self.env.reset()
    return obs

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    obs, reward, done, infos = self.env.step(action)
    return obs, reward, done, infos


# ### Exercise (7 minutes): limit the episode length
# 
# In this exercise, the goal is to create a Gym wrapper that will limit the maximum number of steps per episode (timeout).
# 
# 
# It will also pass a `timeout` signal in the info dict to tell the agent that the termination was due to reaching the limits.

# In[58]:


class TimeLimitWrapper(gym.Wrapper):
  """
  Limit the maximum number of steps per episode.

  :param env: Gym environment that will be wrapped
  :param max_steps: Max number of steps per episode
  """
class TimeLimitWrapper(gym.Wrapper):
  """
  Limit the maximum number of steps per episode.

  :param env: Gym environment that will be wrapped
  :param max_steps: Max number of steps per episode
  """
  def __init__(self, env: gym.Env, max_steps: int = 100):
    # Call the parent constructor, so we can access self.env later
    super().__init__(env)
    self.max_steps = max_steps
    # YOUR CODE HERE
    # Counter of steps per episode
    self.counter=0

    # END OF YOUR CODE
  
  def reset(self) -> GymObs:
    # YOUR CODE HERE
    # TODO: reset the counter and reset the env
    self.counter = 0
    self.env.reset()

    # END OF YOUR CODE
    return obs

  def step(self, action: Union[int, np.ndarray]) -> Tuple[GymObs, float, bool, Dict]:
    # YOUR CODE HERE
    # TODO: 
    # 1. Step into the env
    # 2. Increment the episode counter
    # 3. Overwrite the done signal when time limit is reached 
    # (optional) 4. update the info dict (add a "episode_timeout" key)
    # when the episode was stopped due to timelimit
    obs, reward, done, infos = self.env.step(action)
    self.counter+=1
    if (self.counter >= self.max_steps):
      done=True
      infos['episode_timeout']=True
    # END OF YOUR CODE
    return obs, reward, done, infos


# #### Test the wrapper

# In[59]:


from gym.envs.classic_control.pendulum import PendulumEnv

# Here we create the environment directly because gym.make() already wrap the environement in a TimeLimit wrapper otherwise
env = PendulumEnv()
# Wrap the environment
env = TimeLimitWrapper(env, max_steps=100)


# In[60]:


obs = env.reset()
done = False
n_steps = 0
while not done:
  # Take random actions
  random_action = env.action_space.sample()
  obs, reward, done, infos = env.step(random_action)
  n_steps += 1

print(f"Episode length: {n_steps} steps, info dict: {infos}")


# In practice, `gym` already have a wrapper for that named `TimeLimit` (`gym.wrappers.TimeLimit`) that is used by most environments.

# # Part III: Callbacks
# 
# In this part, you will learn how to use [Callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html) which allow to do monitoring, auto saving, model manipulation, progress bars, ...

# Please read the [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html). Although Stable-Baselines3 provides you with a callback collection (e.g. for creating checkpoints or for evaluation), we are going to re-implement some so you can get a good understanding of how they work.
# 
# To build a custom callback, you need to create a class that derives from `BaseCallback`. This will give you access to events (`_on_training_start`, `_on_step()`) and useful variables (like `self.model` for the RL model).
# 
# `_on_step` returns a boolean value for whether or not the training should continue.
# 
# Thanks to the access to the models variables, in particular `self.model`, we are able to even change the parameters of the model without halting the training, or changing the model's code.

# In[61]:


from stable_baselines3.common.callbacks import BaseCallback


# In[ ]:


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


# Here we have a simple callback that can only be called twice:

# In[62]:


class SimpleCallback(BaseCallback):
    """
    a simple callback that can only be called twice

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(SimpleCallback, self).__init__(verbose)
        self._called = False
    
    def _on_step(self):
      
      if not self._called:
        print("callback - first call")
        self._called = True
        return True # returns True, training continues.

      print("callback - second call")
      return False # returns False, training stops.      


# In[63]:


model = SAC('MlpPolicy', 'Pendulum-v0', verbose=1)
model.learn(8000, callback=SimpleCallback())


# ## Exercise (8 minutes): Checkpoint Callback
# 
# In RL, it is quite useful to save checkpoints during training, as we can end up with burn-in of a bad policy. It also useful if you want to see the progression over time.
# 
# This is a typical use case for callback, as they can call the save function of the model, and observe the training over time.

# In[ ]:


import os

import numpy as np


# In[64]:


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` steps

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose: Whether to print additional infos or not
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        # NOTE: because it derives from `BaseCallback`
        # this checkpoint callback has already access to many variables
        # like `self.model` (cf ``CustomCallback above for a complete list)

    def _init_callback(self) -> None:
        ## YOUR CODE HERE
        # Create folder if needed
        # (you may use `os.makedirs()`)
        os.makedirs(self.save_path, exist_ok=True)

        ## END OF YOUR CODE


    def _on_step(self) -> bool:
        ## YOUR CODE HERE
        # Save the checkpoint if needed
        if (self.num_timesteps % self.save_freq ==0):
          print('save model')
          self.model.save(self.save_path+self.name_prefix+'_'+str(self.num_timesteps))

        ## END OF YOUR CODE
        return True


# Test your callback:

# In[65]:


log_dir = "./tmp/gym/"
# Create Callback
callback = CheckpointCallback(save_freq=1000, save_path="./tmp/gym/", verbose=1)

model = A2C("MlpPolicy", "CartPole-v1", verbose=1)
model.learn(total_timesteps=5000, callback=callback)


# In[66]:


get_ipython().system('ls "./tmp/gym/"')


# Note: The `CheckpointCallback` as well as other [common callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html), like the `EvalCallback` are already included in Stable-Baselines3.

# ## Multiprocessing Demo
# 
# 
# [Vectorized Environments](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html) are a method for stacking multiple independent environments into a single environment. Instead of training an RL agent on 1 environment per step, it allows us to train it on n environments per step. This provides two benefits:
# * Agent experience can be collected more quickly
# * The experience will contain a more diverse range of states, it usually improves exploration
# 
# Stable-Baselines provides two types of Vectorized Environment:
# - SubprocVecEnv which run each environment in a separate process
# - DummyVecEnv which run all environment on the same process
# 
# In practice, DummyVecEnv is usually faster than SubprocVecEnv because of communication delays that subprocesses have.

# In[67]:


import time

from stable_baselines3.common.env_util import make_vec_env


# In[68]:


env = gym.make("Pendulum-v0")
n_steps = 1024


# In[69]:


start_time_one_env = time.time()
model = PPO("MlpPolicy", env, n_epochs=1, n_steps=n_steps, verbose=1).learn(int(2e4))
time_one_env = time.time() - start_time_one_env


# In[70]:


print(f"Took {time_one_env:.2f}s")


# In[71]:


start_time_vec_env = time.time()
# Create 16 environments
vec_env = make_vec_env("Pendulum-v0", n_envs=16)
# At each call to `env.step()`, 16 transitions will be collected, so we account for that for fair comparison
model = PPO("MlpPolicy", vec_env, n_epochs=1, n_steps=n_steps // 16, verbose=1).learn(int(2e4))

time_vec_env = time.time() - start_time_vec_env


# In[72]:


print(f"Took {time_vec_env:.2f}s")


# Note: the speedup is not linear but it is already significant.

# # Part IV: The importance of hyperparameter tuning
# 
# 

# When compared with Supervised Learning, Deep Reinforcement Learning is far more sensitive to the choice of hyper-parameters such as learning rate, number of neurons, number of layers, optimizer ... etc. 
# 
# Poor choice of hyper-parameters can lead to poor/unstable convergence. This challenge is compounded by the variability in performance across random seeds (used to initialize the network weights and the environment).
# 

# ### Challenge (15 minutes): "Grad Student Descent" - Can you beat automatic hyperparameter tuning?
# 
# The challenge is to find the best hyperparameters (max performance) for A2C on `CartPole-v1` with a limited budget of 20 000 training steps.
# 
# You will compete against automatic hyperparameter tuning, good luck ;)
# 
# 
# Maximum reward: 500 on `CartPole-v1`
# 
# The hyperparameters should work for different random seeds.

# In[73]:


budget = int(2e4)


# #### The baseline: default hyperparameters

# In[74]:


model = A2C("MlpPolicy", "CartPole-v1", seed=8, verbose=1).learn(budget)


# In[75]:


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50, deterministic=True)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


# **Your goal is to beat that baseline and get closer to the optimal score of 500**

# Time to tune!

# In[77]:


import torch.nn as nn


# In[78]:


policy_kwargs = dict(
    net_arch=[
      dict(vf=[64, 64], pi=[64, 64]), # network architectures for actor/critic
    ],
    ortho_init=True, # Orthogonal initialization,
    activation_fn=nn.Tanh,
)

hyperparams = dict(
    n_steps=5,
    learning_rate=7e-4,
    gamma=0.9999, # discount factor
    gae_lambda=1.0, # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
                    # Equivalent to classic advantage when set to 1.
    max_grad_norm=0.5, # The maximum value for the gradient clipping
    ent_coef=0.0, # Entropy coefficient for the loss calculation
)

model = A2C("MlpPolicy", "CartPole-v1", seed=8, verbose=1, **hyperparams).learn(budget)


# In[79]:


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50, deterministic=True)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


# Hint - Recommended Hyperparameter Range
# 
# ```python
# gamma = trial.suggest_float("gamma", 0.9, 0.99999, log=True)
# max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
# gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.999, log=True)
# # from 2**3 = 8 to 2**10 = 1024
# n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
# learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
# ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
# ortho_init = trial.suggest_categorical("ortho_init", [False, True])
# # tiny: {"pi": [64], "vf": [64]}
# # default: {"pi": [64, 64], "vf": [64, 64]}
# net_arch = trial.suggest_categorical("net_arch", ["tiny", "default"])
# activation_fn = trial.suggest_categorical("activation_fn", [nn.Tanh, nn.ReLU])
# ```

# Simple example of hyperparameter tuning: https://github.com/optuna/optuna/blob/master/examples/rl/sb3_simple.py
# 
# Complete example: https://github.com/DLR-RM/rl-baselines3-zoo

# # Conclusion
# 
# What we have seen in this notebook:
# - SB3 101
# - Gym wrappers to modify the env
# - SB3 callbacks to access the RL agent
# - multiprocessing to speedup training
# - the importance of good hyperparameters
# - more complete tutorial: https://github.com/araffin/rl-tutorial-jnrr19
# 
# 

# In[ ]:




