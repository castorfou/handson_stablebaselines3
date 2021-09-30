#!/usr/bin/env python
# coding: utf-8

# # Janus gym environment

# In[1]:


import gym
from gym import spaces
import random
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, DDPG
from math import sqrt
import numpy as np
import pandas as pd
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.datasets import make_regression
import warnings
warnings.filterwarnings('ignore') 
import math
import pickle

class Janus(gym.Env):
    metadata = {'render.modes': ['human']}
    template_filename = 'data/dataset-S_public/public/dataset_S-{}.csv'

    def __init__(self, idx=47):
        super(Janus, self).__init__()
        #actions: move on the grid, by continuous value in -1,1
        #0,0 no move
        #based on 94 controlable parameters
        #"We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) ", we will multiply effect by 2
#         self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(94, ))
        #we focus on the 1 most influencal action
        nbr_actions = 4
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(nbr_actions, ))
    

        # all the observation_space
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(228, ))

        file1 = pd.read_csv(self.template_filename.format('file1'), index_col=0)
        file2 = pd.read_csv(self.template_filename.format('file2'), index_col=0)
        file3 = pd.read_csv(self.template_filename.format('file3'), index_col=0)
        vav = pd.read_csv(self.template_filename.format('VAV'), index_col=0)
        self.ti = pd.read_csv(self.template_filename.format('TI'), index_col=0)
        self.ts = pd.read_csv(self.template_filename.format('TS'), index_col=0)
        x_df = file1.copy()
        x_df = x_df.loc[:, (x_df != 0).any(axis=0)]  ## remove static columns
        x_df = x_df.fillna(x_df.mean())  ## replace nan with mean

        self.y_df = file2.copy()
        self.y_df.dropna(how='all', axis=1,
                    inplace=True)  ## drop full tempty columns
        self.y_df = self.y_df.fillna(self.y_df.mean())

        self.vav_df = vav.copy()
        # Dropping few columns
        for dataset in [self.y_df, self.vav_df, self.ti, self.ts]:
            dataset.drop(['target_1', 'target_2', 'target_3', 'target_4'],
                  axis=1,
                  inplace=True)  #to simplify with a 2-dimension target space

        print('features shape: {}, \ntargets shape: {}'.format(
            x_df.shape, self.y_df.shape))

        x_train, x_test, y_train, y_test = train_test_split(x_df,
                                                            self.y_df,
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
        self.full_x = self.full_x.fillna(x_df.mean())
        
        self.partial_x = x_train.copy()

        inferred_y = pd.DataFrame(self.ml_model.predict(self.full_x),
                                  columns=self.y_df.columns)

        # list of [min, max, step, range] for each var
        scale = 100
        decimals = 3
        
        self.list_important_actions = np.argsort(self.ml_model.feature_importances_[:94])[::-1][:nbr_actions]

        ## get limits for Rewards
        self.output_steps = [round((self.y_df[i].max() - self.y_df[i].min())/scale, decimals)                         for i in self.y_df.columns]
        print('Output steps: ', self.output_steps)
        
        self.idx=idx

    def reset(self):
#         self.current_position = self.revert_to_obs_space(
#             self.full_x.sample().values.reshape(-1), self.full_x)
#         random.seed(13)
#         idx = random.randint(0,len(janus_env.partial_x)-1)
        self.current_position = self.revert_to_obs_space(
            self.full_x.iloc[self.idx].values.reshape(-1), self.full_x)
        
        
        # to fix The observation returned by the `reset()` method does not match the given observation space
        # maybe won't happen on linux
        # on windows looks like float64 is the defautl for pandas -> numpy and gym expects float32 (contains tries to cast to dtype(float32))
        self.current_position = self.current_position.astype('float32')
        
        self.last_action = self.current_position[self.list_important_actions]
        self.last_effect = False
        self.global_reward = 0
        self.episode_length = 0
        #print(f'reset at position {self.current_position[:10]}...')
        return self.current_position

    def step(self, action):
#         self.current_position[0:len(action)] = action
        for index, act in enumerate(self.list_important_actions):
            self.current_position[act]=action[index]
        self.last_action = action
        self.episode_length += 1
        done = False
        
        reward = self.reward(
            self.convert_to_real_obs(self.current_position,
                                     self.full_x).values.reshape(1,-1))
        if (reward >= -0.1*self.y_df.shape[1]):
            done = True
        
#         print(f'target reached {done} reward {reward:0.03f} n° step {self.episode_length} action {self.last_action} done threeshold {-0.1*self.y_df.shape[1]:0.03f}')
        
        if done:
            reward += 100
        
        if self.episode_length>100:
            print('episode too long -> reset')
            done = True
            
#         if (max(abs(action))==1):
#             # if on border, we kill episode
#             print('border reached -> done -> reset')
#             reward -= 50
#             done = True
            

        self.global_reward += reward
#         print(f'  reward {reward:0.03f} global reward {self.global_reward:0.03f} done {done} action {action} step {self.episode_length}')
        return self.current_position, reward, done, {}

    def render(self):
        print(
            f'position {self.current_position[:10]}, action {self.last_action[:5]}, effect {self.last_effect}, done {done}, global_reward {self.global_reward:0.03f}'
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

    def reward(self, observation):
        ''' Discrete reward 
        observation if from real world not observation space
        '''

        new_y = self.ml_model.predict(observation).reshape(-1)
#         return self.continuous_reward_clown_hat(new_y)
        return self.reward_smart_archery(new_y)

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
    
    
    
    def continuous_reward_clown_hat(self, new_y):
        ''' Continuous reward '''
        final_reward = 0 

        for i in range(len(new_y)):
            reward = -9
            if ( self.ti.iloc[:,i].values[0] <=  new_y[i] <= self.ts.iloc[:,i].values[0]):
                if ( new_y[i] >= self.vav_df.iloc[:,i].values[0] ):
                    reward = 1-(new_y[i]-self.vav_df.iloc[:,i].values[0])/(self.ts.iloc[:,i].values[0]-self.vav_df.iloc[:,i].values[0])
                else:
                    reward = 1-(self.vav_df.iloc[:,i].values[0]-new_y[i])/(self.vav_df.iloc[:,i].values[0]-self.ti.iloc[:,i].values[0])
            reward += -1
            final_reward+=reward
    #         print(f'reward {reward} final_reward {final_reward} i {i}')

        if (final_reward>0.7*len(new_y)):
            on_target = True
    #         print('On Target : ', new_y)

        return final_reward

    def reward_archery(self, new_y):
        ''' Discrete reward 
        drawback is that you need progress on all targets to get reward improvment
        '''
        final_reward = 0 
        
        ti_target_0 = self.ti.iloc[:,0].values[0]
        ts_target_0 = self.ts.iloc[:,0].values[0]
        ti_target_5 = self.ti.iloc[:,1].values[0]
        ts_target_5 = self.ts.iloc[:,1].values[0]
        x, y = new_y[0], new_y[1]
        if ( (ti_target_0*0.10 <= x <= ts_target_0*0.10) & ( ti_target_5*0.10 <= y <= ts_target_5*0.10 )):
            reward = 0
        else:
            if ( (ti_target_0*0.50 <= x <= ts_target_0*0.50) & ( ti_target_5*0.50 <= y <= ts_target_5*0.50 )):
                reward = -2
            else:
                if ( (ti_target_0 <= x <= ts_target_0) & ( ti_target_5 <= y <= ts_target_5 )):
                    reward = -5
                else:
                    reward = -20
        final_reward = reward
        return final_reward

    def reward_smart_archery(self, new_y):
        ''' Discrete reward '''
        ti_target_0 = self.ti.iloc[:,0].values[0]
        ts_target_0 = self.ts.iloc[:,0].values[0]
        ti_target_5 = self.ti.iloc[:,1].values[0]
        ts_target_5 = self.ts.iloc[:,1].values[0]
        x, y = new_y[0], new_y[1]
        
        reward_x = 0
        if (x <= ti_target_0 or x >= ts_target_0): reward_x = -10
        if ( ti_target_0 <= x <= 0.5*ti_target_0  or 0.5*ts_target_0 <= x <= ts_target_0  ): reward_x = -3
        if ( 0.5*ti_target_0 <= x <= 0.1*ti_target_0  or 0.1*ts_target_0 <= x <= 0.5*ts_target_0  ): reward_x = -1
        reward_y = 0
        if (y <= ti_target_5 or y >= ts_target_5): reward_y = -10
        if ( ti_target_5 <= y <= 0.5*ti_target_5  or 0.5*ts_target_5 <= y <= ts_target_5  ): reward_y = -3
        if ( 0.5*ti_target_5 <= y <= 0.1*ti_target_5  or 0.1*ts_target_5 <= y <= 0.5*ts_target_5  ): reward_y = -1
                    
        final_reward = reward_x + reward_y
        return final_reward
    


# ## env check

# In[2]:


from stable_baselines3.common.env_checker import check_env

janus_env = Janus()

check_env(janus_env)


# # Train RL model

# In[ ]:


from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import warnings
warnings.filterwarnings('ignore') 

model_name = "EXP10 - IDX 11927 - smart archery"

idx = 11927

janus = Janus(idx)
check_env(janus)

n_actions = janus.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model_janus_td3 = TD3("MlpPolicy", janus, action_noise=action_noise, verbose=2,tensorboard_log="./tensorboard/")
model_janus_td3.learn(total_timesteps=10000, log_interval=4, tb_log_name=model_name)
model_janus_td3.save("./data/"+model_name)


# # Reward functions

# ## calculate reward outputs

# In[15]:


janus_env = Janus()
inf_x = janus_env.ti.iloc[0][0]
sup_x = janus_env.ts.iloc[0][0]
inf_y = janus_env.ti.iloc[0][1]
sup_y = janus_env.ts.iloc[0][1]
inf_limx = min(inf_x, janus_env.y_df.min()[0])
sup_limx = max(sup_x, janus_env.y_df.max()[0])
inf_limy = min(inf_y, janus_env.y_df.min()[1])
sup_limy = max(sup_y, janus_env.y_df.max()[1])

def calculate_reward_outputs(reward=janus_env.continuous_reward_clown_hat, reward_name = 'janus_env.continuous_reward_clown_hat'):
    '''
    take a grid of all possible values for target_0, target_5 (between [min(target) - 3, max(target)+3]
    and calculate reward values
    keep that in a dataframe with columns ['x', 'y', 'z']
    at /data/reward_<reward_name>.csv
    '''

    X = np.arange(inf_limx-3, sup_limx+3, 0.2)
    Y = np.arange(inf_limy-3, sup_limy+3, 0.2)
    xyz_content=[]
    for x in X:
        for y in Y:
            xyz_content.append([x, y, reward([x,y])])
    xyz_content = pd.DataFrame(xyz_content, columns=['x', 'y', 'z'])
    xyz_content.to_csv('./data/reward_'+reward_name+'.csv')


# In[14]:


calculate_reward_outputs(janus_env.continuous_reward_clown_hat, 'clown_hat')


# ## plot reward

# In[10]:


import plotly.express as px

def plot_reward(csv_reward_outputs):    
    xyz_content = pd.read_csv('./data/'+csv_reward_outputs, index_col=0)
    fig = px.scatter_3d(xyz_content, x='x', y='y', z='z',
                  color='z')
    fig.show()


# In[11]:


plot_reward('reward_clown_hat.csv')


# # Apply reward to all dropouts

# In[ ]:




