#!/usr/bin/env python
# coding: utf-8

# # Janus gym environment

# In[10]:


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
        return self.reward_archery(new_y)

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
        ''' Continuous reward '''
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


# In[6]:


janus_env = Janus()
from stable_baselines3.common.env_checker import check_env

check_env(janus_env)


# # SAC training

# In[127]:


from stable_baselines3 import SAC

janus_env = Janus()
check_env(janus_env)

model_janus_sac = SAC("MlpPolicy", janus_env, verbose=2,tensorboard_log="./tensorboard/")
model_janus_sac.learn(total_timesteps=100000, log_interval=4, tb_log_name="janus partial sac")

janus_env.reset()
for i in range(100):
    action, _ = model_janus_sac.predict(janus_env.current_position)
    print(f'action {action}')
    obs, rewards, done, info = janus_env.step(action)
    janus_env.render()
    if done: break
janus_env.close()


# # TD3 training

# In[63]:


from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

model_name = "janus partial td3 - reward clown hat noborder"

janus_env = Janus()
check_env(janus_env)

n_actions = janus_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model_janus_td3 = TD3("MlpPolicy", janus_env, action_noise=action_noise, verbose=2,tensorboard_log="./tensorboard/")
model_janus_td3.learn(total_timesteps=1000, log_interval=4, tb_log_name=model_name)
model_janus_td3.save("./data/"+model_name)


# In[101]:


def use_trained_RL_model():
    print(f'model used: {model_name}')

    model_janus_td3 = TD3("MlpPolicy", janus_env, action_noise=action_noise, verbose=2,tensorboard_log="./tensorboard/")
    model_janus_td3.load("./data/"+model_name)

    janus_env.reset()
    for i in range(100):
        action, _ = model_janus_td3.predict(janus_env.current_position)
        print(f'action {action}')
        obs, rewards, done, info = janus_env.step(action)
        janus_env.render()
        if done: 
            print(f'done within {i+1} iterations')
            break
    janus_env.close()


# In[104]:


def use_trained_RL_model_2(idx):
    janus_env = Janus(idx)
    check_env(janus_env)

    # model_name = "janus partial td3 - reward clown hat"
    print(f'model used: {model_name}')
    model_janus_td3 = TD3("MlpPolicy", janus_env, action_noise=action_noise, verbose=2,tensorboard_log="./tensorboard/")
    model_janus_td3.load("./data/"+model_name)

    janus_env.reset()
    new_y = janus_env.ml_model.predict(janus_env.convert_to_real_obs(janus_env.current_position,
                                         janus_env.full_x).values.reshape(1,-1)).reshape(-1)
    print(f'init results:  {new_y} reward {janus_env.continuous_reward_clown_hat(new_y):0.03f} action {janus_env.last_action}')

    for i in range(100):
        action, _ = model_janus_td3.predict(janus_env.current_position)
    #     print(f'action {action}')
        obs, rewards, done, info = janus_env.step(action)
    #     janus_env.render()
        if done: 
            print(f'done within {i+1} iterations')
            break
    new_y = janus_env.ml_model.predict(janus_env.convert_to_real_obs(janus_env.current_position,
                                         janus_env.full_x).values.reshape(1,-1)).reshape(-1)
    print(f'new results after {i+1} iterations:  {new_y} associated reward {janus_env.continuous_reward_clown_hat(new_y):0.03f}  action {janus_env.last_action}')
    return janus_env.vav_df


# In[105]:


use_trained_RL_model_2(2230)


# # plot reward functions

# In[66]:


janus_env = Janus()
# print(janus_env.vav_df)
# print(janus_env.ti)
# print(janus_env.ts)

inf_x = janus_env.ti.iloc[0][0]
sup_x = janus_env.ts.iloc[0][0]
inf_y = janus_env.ti.iloc[0][1]
sup_y = janus_env.ts.iloc[0][1]
inf_limx = min(inf_x, janus_env.y_df.min()[0])
sup_limx = max(sup_x, janus_env.y_df.max()[0])
inf_limy = min(inf_y, janus_env.y_df.min()[1])
sup_limy = max(sup_y, janus_env.y_df.max()[1])


# ## create dataframe with reward values

# In[8]:


def keep_reward_content(reward=janus_env.continuous_reward_clown_hat, reward_name = 'janus_env.continuous_reward_clown_hat'):

    X = np.arange(inf_limx-3, sup_limx+3, 0.2)
    Y = np.arange(inf_limy-3, sup_limy+3, 0.2)
    xyz_content=[]
    for x in X:
        for y in Y:
            xyz_content.append([x, y, reward([x,y])])
    xyz_content = pd.DataFrame(xyz_content, columns=['x', 'y', 'z'])
    xyz_content.to_csv('./data/'+reward_name+'.csv')
        


# In[22]:


keep_reward_content(janus_env.continuous_reward_clown_hat, 'continuous_reward_clown_hat')


# ## plot reward function

# In[12]:


import plotly.express as px

xyz_content = pd.read_csv('./data/continuous_reward_clown_hat.csv', index_col=0)
fig = px.scatter_3d(xyz_content, x='x', y='y', z='z',
              color='z')
fig.show()


# # visualiser les progrès par RL: qualité avant et après

# In[78]:


janus_env = Janus()
janus_env.reset()
inf_x = janus_env.ti.iloc[0][0]
sup_x = janus_env.ts.iloc[0][0]
inf_y = janus_env.ti.iloc[0][1]
sup_y = janus_env.ts.iloc[0][1]
inf_limx = min(inf_x, janus_env.y_df.min()[0])
sup_limx = max(sup_x, janus_env.y_df.max()[0])
inf_limy = min(inf_y, janus_env.y_df.min()[1])
sup_limy = max(sup_y, janus_env.y_df.max()[1])
vav_x = janus_env.vav_df.iloc[0].values[0]
vav_y = janus_env.vav_df.iloc[0].values[1]


# In[111]:


def positionne_point_idx(janus_env, fig):
    print(f'Index de l observation {janus_env.idx}')
    current_position = janus_env.revert_to_obs_space(janus_env.full_x.iloc[janus_env.idx].values.reshape(-1), janus_env.full_x)
    current_position = current_position.astype('float32')
    current_y = janus_env.ml_model.predict(janus_env.convert_to_real_obs(current_position,
                                     janus_env.full_x).values.reshape(1,-1)).reshape(-1)
    print(f'init results:  {current_y} reward {janus_env.continuous_reward_clown_hat(current_y):0.03f} action {janus_env.last_action}')
    fig.add_trace(go.Scatter(x=[current_y[0]], y=[current_y[1]], name='point initial obs '+str(janus_env.idx)))

    
    for i in range(100):
        action, _ = model_janus_td3.predict(janus_env.current_position)
    #     print(f'action {action}')
        obs, rewards, done, info = janus_env.step(action)
    #     janus_env.render()
        if done: 
            print(f'done within {i+1} iterations')
            break
    new_y = janus_env.ml_model.predict(janus_env.convert_to_real_obs(janus_env.current_position,
                                         janus_env.full_x).values.reshape(1,-1)).reshape(-1)
    
    print(f'final results:  {new_y} reward {janus_env.continuous_reward_clown_hat(new_y):0.03f} action {janus_env.last_action}')
    fig.add_trace(go.Scatter(x=[new_y[0]], y=[new_y[1]], name='point final obs '+str(janus_env.idx)))
    
    


# In[112]:


import plotly.graph_objects as go


def visualise_avant_apres(janus_env):
    fig = go.Figure()

    fig.update_xaxes(title_text='target_0', range=[inf_limx-3, sup_limx+3])
    fig.update_yaxes(title_text='target_5', range=[inf_limy-3, sup_limy+3])

    fig.add_trace(go.Scatter(x=[inf_x,inf_x,sup_x,sup_x, inf_x, None, vav_x], y=[inf_y,sup_y,sup_y, inf_y, inf_y, None, vav_y], fill="toself", name='TI, TS, VAV'))
    positionne_point_idx(janus_env, fig)
    fig.update_layout({'showlegend': True, 'title':'Position de la qualité, avant, après'})
    fig.show()


# # visualiser les reward de toutes les tombées

# ## toutes les tombées (13639)

# In[93]:


#13000 tombees
janus_env.full_x
#782 tombees MGQA
janus_env.y_df

full_prediction = janus_env.ml_model.predict(janus_env.full_x)
full_prediction
reward_df = pd.DataFrame(full_prediction, columns=['target_0', 'target_5'])
reward_df

reward_df['reward']=reward_df.apply(lambda x: janus_env.continuous_reward_clown_hat([x[0], x[1]]), axis=1)


# In[94]:


reward_df


# In[95]:


import pandas as pd
pd.options.plotting.backend = "plotly"

fig = reward_df.plot()
fig.show()


# # EXP 6 - IDX 2230

# In[97]:


janus_env_2230 = Janus(2230)


# In[100]:


from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import warnings
warnings.filterwarnings('ignore') 

model_name = "EXP6 - IDX 2230 - reward cont clown"

janus_env_2230 = Janus(2230)
check_env(janus_env_2230)

n_actions = janus_env_2230.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model_janus_td3 = TD3("MlpPolicy", janus_env_2230, action_noise=action_noise, verbose=2,tensorboard_log="./tensorboard/")
model_janus_td3.learn(total_timesteps=10000, log_interval=4, tb_log_name=model_name)
model_janus_td3.save("./data/"+model_name)


# In[114]:


janus_env_2230 = Janus(2230)
janus_env_2230.reset()
print(f'model used: {model_name}')
model_janus_td3 = TD3("MlpPolicy", janus_env, action_noise=action_noise, verbose=2,tensorboard_log="./tensorboard/")
model_janus_td3.load("./data/"+model_name)


# In[115]:


visualise_avant_apres(janus_env_2230)


# # EXP 7 - IDX 11926

# In[116]:


from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import warnings
warnings.filterwarnings('ignore') 

model_name = "EXP7 - IDX 11926 - reward cont clown"

idx = 11926

janus = Janus(idx)
check_env(janus)

n_actions = janus.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model_janus_td3 = TD3("MlpPolicy", janus, action_noise=action_noise, verbose=2,tensorboard_log="./tensorboard/")
model_janus_td3.learn(total_timesteps=10000, log_interval=4, tb_log_name=model_name)
model_janus_td3.save("./data/"+model_name)


# In[117]:


visualise_avant_apres(janus)


# # move to reward archery and plot

# On change de 
# 
# ```python
#     def reward(self, observation):
#         ''' Discrete reward 
#         observation if from real world not observation space
#         '''
#         new_y = self.ml_model.predict(observation).reshape(-1)
#         return self.continuous_reward_clown_hat(new_y)
# ```
# 
# à
# ```python
#     def reward(self, observation):
#         ''' Discrete reward 
#         observation if from real world not observation space
#         '''
#         new_y = self.ml_model.predict(observation).reshape(-1)
#         return self.reward_archery(new_y)
# ```
# 
# 

# In[11]:


janus_env = Janus()
# print(janus_env.vav_df)
# print(janus_env.ti)
# print(janus_env.ts)

inf_x = janus_env.ti.iloc[0][0]
sup_x = janus_env.ts.iloc[0][0]
inf_y = janus_env.ti.iloc[0][1]
sup_y = janus_env.ts.iloc[0][1]
inf_limx = min(inf_x, janus_env.y_df.min()[0])
sup_limx = max(sup_x, janus_env.y_df.max()[0])
inf_limy = min(inf_y, janus_env.y_df.min()[1])
sup_limy = max(sup_y, janus_env.y_df.max()[1])

keep_reward_content(janus_env.reward_archery, 'reward_archery')

import plotly.express as px

xyz_content = pd.read_csv('./data/reward_archery.csv', index_col=0)
fig = px.scatter_3d(xyz_content, x='x', y='y', z='z',
              color='z')
fig.show()


# # EXP 8 - archery - IDX  11926

# In[13]:


from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import warnings
warnings.filterwarnings('ignore') 

model_name = "EXP8 - IDX 11926 - archery"

idx = 11926

janus = Janus(idx)
check_env(janus)

n_actions = janus.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model_janus_td3 = TD3("MlpPolicy", janus, action_noise=action_noise, verbose=2,tensorboard_log="./tensorboard/")
model_janus_td3.learn(total_timesteps=10000, log_interval=4, tb_log_name=model_name)
model_janus_td3.save("./data/"+model_name)


# visu reward archery toutes les tombées

# In[14]:


full_prediction = janus.ml_model.predict(janus.full_x)
full_prediction
reward_df = pd.DataFrame(full_prediction, columns=['target_0', 'target_5'])
reward_df

reward_df['reward_archery']=reward_df.apply(lambda x: janus.reward_archery([x[0], x[1]]), axis=1)


# In[16]:


reward_df.head()


# In[17]:


import pandas as pd
pd.options.plotting.backend = "plotly"

fig = reward_df.plot()
fig.show()


# # Fix notebook duplicated cells

# In[29]:


import nbformat as nbf
from glob import glob

import uuid
def get_cell_id(id_length=8):
    return uuid.uuid4().hex[:id_length]

# your notebook name/keyword
nb_name = '03 - partial environnement for dataset S.ipynb'
notebooks = list(filter(lambda x: nb_name in x, glob("./*.ipynb", recursive=True)))

# iterate over notebooks
for ipath in sorted(notebooks):
    # load notebook
    ntbk = nbf.read(ipath, nbf.NO_CONVERT)
    
    cell_ids = []
    for cell in ntbk.cells:
        cell_ids.append(cell['id'])

    # reset cell ids if there are duplicates
    if not len(cell_ids) == len(set(cell_ids)): 
        for cell in ntbk.cells:
            cell['id'] = get_cell_id()

    nbf.write(ntbk, ipath)


# # EXP 9 - archery - IDX 1926

# In[30]:


from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import warnings
warnings.filterwarnings('ignore') 

model_name = "EXP9 - IDX 1926 - archery"

idx = 1926

janus = Janus(idx)
check_env(janus)

n_actions = janus.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model_janus_td3 = TD3("MlpPolicy", janus, action_noise=action_noise, verbose=2,tensorboard_log="./tensorboard/")
model_janus_td3.learn(total_timesteps=10000, log_interval=4, tb_log_name=model_name)
model_janus_td3.save("./data/"+model_name)


# In[32]:


idx = 1926

janus = Janus(idx)
janus.reset()
visualise_avant_apres(janus)


# In[ ]:




