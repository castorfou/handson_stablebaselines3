#!/usr/bin/env python
# coding: utf-8

# # Janus temporary gym environment

# In[1]:


import gym
from gym import spaces
import random
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, DDPG
from math import sqrt
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


class JanusTemp(gym.Env):
    metadata = {'render.modes': ['human']}
    template_filename = 'data/dataset-S_public/public/dataset_S-{}.csv'




    def __init__(self):
        super(JanusTemp, self).__init__()
        #actions: move on the grid, by continuous value in -1,1
        #0,0 no move
        #based on 94 controlable parameters
        #"We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) ", we will multiply effect by 2
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(98, ))

        # all the observation_space
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(325, ))
        

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
        if ((action[0] + self.current_position[1] < 5) &
            (action[0] + self.current_position[1] > 0)):
            effect = True
            self.current_position[1] += action[0]
        if ((action[1] + self.current_position[0] < 5) &
            (action[1] + self.current_position[0] > 0)):
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
        print(
            f'position {self.current_position}, action {self.last_action}, effect {self.last_effect}, done {done}, global_reward {self.global_reward}'
        )

    def convert_to_real_obs(self, observation, observation_dataset):
        '''
        to convert an observation from observation space ([-1, 1],325) to  real world
        -1 matches with min() of each column
        1 matches with max() of each column
        
        observation: instance of observation_space
        observation_dataset: the full real dataset (obfuscated in that case)
        '''
        return (observation + np.ones(self.observation_space.shape)) / 2 * (
            observation_dataset.max() -
            observation_dataset.min()) + observation_dataset.min()

    def revert_to_obs_space(self, real_observation, observation_dataset):
        '''
        to revert an observation sample (from real world) to observation space
        min() of each column will match with -1
        max() of each column will match with +1
        
        real_observation: instance of real_world
        observation_dataset: the full real dataset (obfuscated in that case)
        '''
        return np.nan_to_num(
            2 * (real_observation - observation_dataset.min()) /
            (observation_dataset.max() - observation_dataset.min()) -
            np.ones(self.observation_space.shape)).reshape(-1)


# # load of data

# In[2]:


import numpy as np
import pandas as pd

template_filename = 'data/dataset-S_public/public/dataset_S-{}.csv'


file1 = pd.read_csv(template_filename.format('file1'), index_col=0)
file2 = pd.read_csv(template_filename.format('file2'), index_col=0)
file3 = pd.read_csv(template_filename.format('file3'), index_col=0)
vav = pd.read_csv(template_filename.format('VAV'), index_col=0)
ti = pd.read_csv(template_filename.format('TI'), index_col=0)
ts = pd.read_csv(template_filename.format('TS'), index_col=0)


# In[3]:


print(f'valeur min dans le dataset {min(file3.min())}, \ntop 5 min \n {file3.min()[file3.min().argsort()].head(5)}')
print(f'valeur maxi dans le dataset {max(file3.max())}, \ntop 5 max \n {file3.max()[file3.max().argsort(reversed)].tail(5)}')


# In[4]:


file3.describe()


# ## convert observation space sample to real observation 

# In[5]:


def convert_to_real_obs(observation, observation_dataset):
    '''
    to convert an observation from observation space ([-1, 1],325) to  real world
    -1 matches with min() of each column
    1 matches with max() of each column

    observation: instance of observation_space
    observation_dataset: the full real dataset (obfuscated in that case)
    '''
    return (observation + np.ones(len(observation))) / 2 * (
        observation_dataset.max() -
        observation_dataset.min()) + observation_dataset.min()


# In[6]:


env_test = JanusTemp()
from stable_baselines3.common.env_checker import check_env

# check_env(env_test)

convert_to_real_obs(env_test.observation_space.sample(), file3)


# In[7]:


obs = env_test.observation_space.sample()
arr = convert_to_real_obs(obs, file3)

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Observation in real world (bottom), derived from observation space (top)')
ax2.plot(np.arange(0, len(arr)), arr)
ax2.set_ylim([-20,20])
ax1.plot(np.arange(0, len(arr)), obs)
plt.show()


# ## revert 

# In[8]:


def revert_to_obs_space(real_observation, observation_dataset):
    '''
    to revert an observation sample (from real world) to observation space
    min() of each column will match with -1
    max() of each column will match with +1

    real_observation: instance of real_world
    observation_dataset: the full real dataset (obfuscated in that case)
    '''
    return np.nan_to_num(2 * (real_observation - observation_dataset.min()) / (
        observation_dataset.max() - observation_dataset.min()) - np.ones(
            real_observation.shape[1])).reshape(-1)


# In[9]:


revert_to_obs_space(file3.sample(), file3)


# # prediction and reward calculation

# ## preprocessing, train, test data

# In[10]:


## ML model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.datasets import make_regression

import math
import pickle


x_df = file1.copy()
x_df = x_df.loc[:, (x_df != 0).any(axis=0)]    ## remove static columns 
x_df = x_df.fillna(x_df.mean())                ## replace nan with mean

y_df = file2.copy()
y_df.dropna(how='all', axis=1, inplace=True)   ## drop full tempty columns
y_df = y_df.fillna(y_df.mean())

vav_df = vav.copy()
# Dropping few columns

for dataset in [y_df, vav_df, ti, ts]:
    dataset.drop(['target_1', 'target_2', 'target_3', 'target_4'],
          axis=1,
          inplace=True)  #to simplify with a 2-dimension target space


print('features shape: {}, \ntargets shape: {}'.format(x_df.shape, y_df.shape))

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.1, random_state=14)
print('\nLength of train is {}, test is {}'.format(len(x_train), len(x_test)))


# In[11]:


action_columns = [col for col in x_df.columns if int(col.split('_')[-1])<=97] ##data_97 is the last Action column
print(f'Action space dimension: {len(action_columns)}')
print(f'Observation space dimension: {len(x_df.columns)}')


# ## rf prediction model

# In[12]:


## Model fitting

ml_model = RandomForestRegressor()   
ml_model.fit(x_train, y_train)

test_pred = ml_model.predict(x_test)

mse = mean_squared_error(y_test, test_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, test_pred)       

print('MSE: ', mse)
print('RMSE: ',rmse)
print('R-squared: ',r2)


# In[13]:


## ---- save/load the model to disk

## Random forest
filename = 'data/models/janus_RF.pkl'  # janus_LinearReg, janus_RF

# pickle.dump(ml_model, open(filename, 'wb'))

# # load the model from disk
ml_model = pickle.load(open(filename, 'rb'))
print(f'R squared: {ml_model.score(x_test, y_test.values):0.04f}')


# ## reward calculation

# In[14]:


# list of [min, max, step, range] for each var
scale = 100
decimals = 3

## get limits for Rewards
output_steps = [round((y_df[i].max() - y_df[i].min())/scale, decimals)                 for i in y_df.columns]
print('Output steps: ', output_steps)


def discrete_reward(new_y):
    ''' Discrete reward '''
    
    new_val = [math.sqrt((vav_df.iloc[:,i].values[0] - new_y[i])**2) for i in range(len(new_y))]
    k = 10
    k1 = 1
    if new_val[0] < k * output_steps[0] and new_val[1] < k * output_steps[1]:
        reward = 1 #dans les 10% d'amplitude max autour de la vav
        if new_val[0] < k1 * output_steps[0] and new_val[
                1] < k1 * output_steps[1]:
            reward = 10 #dans les 1% d'amplitude max autour de la vav
            on_target = True
            print('On Target : ', new_y)
    else:
        reward = -1
    return reward

#vav
print(discrete_reward([vav['target_0'].values[0], vav['target_5'].values[0]]))
#vav + 1%/2
print(discrete_reward([vav['target_0'].values[0]+output_steps[0]/2, vav['target_5'].values[0]+output_steps[0]/2]))
#vav + 5%
print(discrete_reward([vav['target_0'].values[0]+5*output_steps[0], vav['target_5'].values[0]+5*output_steps[0]]))
#vav + 10%
print(discrete_reward([vav['target_0'].values[0]+10*output_steps[0], vav['target_5'].values[0]+10*output_steps[0]]))


# ![image.png](attachment:image.png)

# In[50]:


def discrete_reward_continuous(new_y):
    ''' Discrete reward '''
    final_reward = 0 
    
    for i in range(len(new_y)):
        reward = -5
        if ( ti.iloc[:,i].values[0] <=  new_y[i] <= ts.iloc[:,i].values[0]):
            if ( new_y[i] >= vav_df.iloc[:,i].values[0] ):
                reward = 1-(new_y[i]-vav_df.iloc[:,i].values[0])/(ts.iloc[:,i].values[0]-vav_df.iloc[:,i].values[0])
            else:
                reward = 1-(new_y[i]-ti.iloc[:,i].values[0])/(vav_df.iloc[:,i].values[0]-ti.iloc[:,i].values[0])
        final_reward+=reward
#         print(f'reward {reward} final_reward {final_reward} i {i}')

    if (final_reward>0.7*len(new_y)):
        on_target = True
#         print('On Target : ', new_y)
        
    return final_reward

#vav
print(discrete_reward_continuous([vav['target_0'].values[0], vav['target_5'].values[0]]))
print(discrete_reward_continuous([vav['target_0'].values[0]+0.1, vav['target_5'].values[0]]))


# In[48]:


import matplotlib.pyplot as plt


def plot_targets_reward(dataset):
    targets = list(dataset.columns)
    nbr_targets = len(targets)
    fig, axes = plt.subplots(nbr_targets + 1, 1)
    fig.suptitle(str(targets) + ', reward')
    fig.set_size_inches(18.5, 10.5)
    for count, ax in enumerate(axes[:-1]):
        ax.plot(np.arange(0, len(dataset[targets[count]])),
                dataset[targets[count]],
                label=targets[count])
        ax.axhline(y=vav_df[targets[count]].values[0],
                   color='k',
                   lw=0.8,
                   ls='--',
                   label='vav')
        ax.axhline(y=ti[targets[count]].values[0],
                   color='r',
                   lw=0.8,
                   ls='--',
                   label='ti')
        ax.axhline(y=ts[targets[count]].values[0],
                   color='r',
                   lw=0.8,
                   ls='--',
                   label='ts')
        
        
        ax.legend()
    rewards = [discrete_reward_continuous([dataset.loc[i, targets[0]], dataset.loc[i, targets[1]]]) for i in dataset.index]
    axes[-1].plot(np.arange(0, len(dataset[targets[count]])),
                  rewards, label='reward')
    axes[-1].legend()
    print(f'Mean of reward: {np.mean(rewards)}')
    plt.show()


# In[49]:


plot_targets_reward(y_df)


# In[18]:


full_x = file3.copy()[x_df.columns]
full_x = full_x.fillna(x_df.mean()) 

inferred_y = pd.DataFrame(ml_model.predict(full_x), columns=y_df.columns)
plot_targets_reward(inferred_y)


# In[19]:


partial_x = file1.copy()[x_df.columns]
partial_x = partial_x.fillna(x_df.mean())

predicted_partial = pd.DataFrame(ml_model.predict(partial_x), columns=y_df.columns)
plot_targets_reward(predicted_partial)


# ## reward calculation based on observation

# In[20]:


real_observation = full_x.sample()

print(f'real observation (sampled from full_x: {real_observation}')
predicted_y_from_obs = ml_model.predict(real_observation).reshape(-1)
print(f'predicted y from this sample {predicted_y_from_obs} {predicted_y_from_obs.shape}')
discrete_reward(predicted_y_from_obs)


# In[21]:


# list of [min, max, step, range] for each var
scale = 100
decimals = 3

## get limits for Rewards
output_steps = [round((y_df[i].max() - y_df[i].min())/scale, decimals)                 for i in y_df.columns]

def discrete_reward_from_obs(observation):
    ''' Discrete reward 
    observation if from real world not observation space
    '''
    
    new_y = ml_model.predict(observation).reshape(-1)
    return discrete_reward(new_y)

print(discrete_reward_from_obs(full_x.sample()))


# In[22]:


for i in range(100):
    print(discrete_reward_from_obs(full_x.sample()))


# # Janus gym environment

# In[23]:


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
    template_filename = 'data/dataset-S_public/public/dataset_S-{}.csv'

    def __init__(self):
        super(Janus, self).__init__()
        #actions: move on the grid, by continuous value in -1,1
        #0,0 no move
        #based on 94 controlable parameters
        #"We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) ", we will multiply effect by 2
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(94, ))

        # all the observation_space
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(228, ))

        file1 = pd.read_csv(template_filename.format('file1'), index_col=0)
        file2 = pd.read_csv(template_filename.format('file2'), index_col=0)
        file3 = pd.read_csv(template_filename.format('file3'), index_col=0)
        vav = pd.read_csv(template_filename.format('VAV'), index_col=0)
        x_df = file1.copy()
        x_df = x_df.loc[:, (x_df != 0).any(axis=0)]  ## remove static columns
        x_df = x_df.fillna(x_df.mean())  ## replace nan with mean

        y_df = file2.copy()
        y_df.dropna(how='all', axis=1,
                    inplace=True)  ## drop full tempty columns
        y_df = y_df.fillna(y_df.mean())

        self.vav_df = vav.copy()
        # Dropping few columns
        y_df.drop(['target_1', 'target_2', 'target_3', 'target_4'],
                  axis=1,
                  inplace=True)  #to simplify with a 2-dimension target space
        self.vav_df.drop(['target_1', 'target_2', 'target_3', 'target_4'],
                         axis=1,
                         inplace=True)

        print('features shape: {}, \ntargets shape: {}'.format(
            x_df.shape, y_df.shape))

        x_train, x_test, y_train, y_test = train_test_split(x_df,
                                                            y_df,
                                                            test_size=0.1,
                                                            random_state=14)
        print('\nLength of train is {}, test is {}'.format(
            len(x_train), len(x_test)))
        ## Random forest
        filename = 'data/models/janus_RF.pkl'  # janus_LinearReg, janus_RF

        # pickle.dump(ml_model, open(filename, 'wb'))

        # # load the model from disk
        self.ml_model = pickle.load(open(filename, 'rb'))
        print(f'R squared: {self.ml_model.score(x_test, y_test.values):0.04f}')

        self.full_x = file3.copy()[x_df.columns]
        self.full_x = full_x.fillna(x_df.mean())

        inferred_y = pd.DataFrame(ml_model.predict(full_x),
                                  columns=y_df.columns)

        # list of [min, max, step, range] for each var
        scale = 100
        decimals = 3

        ## get limits for Rewards
        self.output_steps = [round((y_df[i].max() - y_df[i].min())/scale, decimals)                         for i in y_df.columns]
        print('Output steps: ', output_steps)

    def reset(self):
        self.current_position = self.revert_to_obs_space(
            self.full_x.sample().values.reshape(-1), self.full_x)
        self.last_action = np.array([])
        self.last_effect = False
        self.global_reward = 0
        self.episode_length = 0
        #print(f'reset at position {self.current_position[:10]}...')
        return self.current_position

    def step(self, action):
        self.current_position[0:len(action)] = action
        self.last_action = action
        self.episode_length += 1
        
        reward = self.discrete_reward_from_obs(
            self.convert_to_real_obs(self.current_position,
                                     self.full_x).values.reshape(1,-1))
        done = reward == 10
        
        if self.episode_length>100:
            #print('episode too long -> reset')
            done = True
            

        self.global_reward += reward
        return self.current_position, reward, done, {}

    def render(self):
        print(
            f'position {self.current_position[:10]}, action {self.last_action[:5]}, effect {self.last_effect}, done {done}, global_reward {self.global_reward}'
        )

    def convert_to_real_obs(self, observation, observation_dataset):
        '''
        to convert an observation from observation space ([-1, 1],325) to  real world
        -1 matches with min() of each column
        1 matches with max() of each column
        
        observation: instance of observation_space
        observation_dataset: the full real dataset (obfuscated in that case)
        '''
        return (observation + np.ones(self.observation_space.shape)) / 2 * (
            observation_dataset.max() -
            observation_dataset.min()) + observation_dataset.min()

    def revert_to_obs_space(self, real_observation, observation_dataset):
        '''
        to revert an observation sample (from real world) to observation space
        min() of each column will match with -1
        max() of each column will match with +1
        
        real_observation: instance of real_world
        observation_dataset: the full real dataset (obfuscated in that case)
        '''
        return np.nan_to_num(
            2 * (real_observation - observation_dataset.min()) /
            (observation_dataset.max() - observation_dataset.min()) -
            np.ones(self.observation_space.shape)).reshape(-1)

    def discrete_reward_from_obs(self, observation):
        ''' Discrete reward 
        observation if from real world not observation space
        '''

        new_y = self.ml_model.predict(observation).reshape(-1)
        return self.discrete_reward(new_y)

    def discrete_reward(self, new_y):
        ''' Discrete reward '''

        new_val = [
            sqrt((self.vav_df.iloc[:, i].values[0] - new_y[i])**2)
            for i in range(len(new_y))
        ]
        k = 10
        k1 = 1
        if new_val[0] < k * self.output_steps[0] and new_val[
                1] < k * self.output_steps[1]:
            reward = 1  #dans les 10% d'amplitude max autour de la vav
            if new_val[0] < k1 * self.output_steps[0] and new_val[
                    1] < k1 * self.output_steps[1]:
                reward = 10  #dans les 1% d'amplitude max autour de la vav
                on_target = True


#                 print('On Target : ', new_y)
        else:
            reward = -1
        return reward


# In[24]:


janus_env = Janus()
from stable_baselines3.common.env_checker import check_env

check_env(janus_env)


# In[ ]:





# # SAC training

# In[99]:


from stable_baselines3 import SAC

janus_env = Janus()
check_env(janus_env)

model_janus_sac = SAC("MlpPolicy", janus_env, verbose=2,tensorboard_log="./tensorboard/")
model_janus_sac.learn(total_timesteps=100000, log_interval=4, tb_log_name="janus sac")

janus_env.reset()
for i in range(100):
    action, _ = model_janus_sac.predict(janus_env.current_position)
    print(f'action {action}')
    obs, rewards, done, info = janus_env.step(action)
    janus_env.render()
    if done: break
janus_env.close()


# In[ ]:




