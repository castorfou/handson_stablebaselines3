#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from exp.janus_rl import *


# # reward smart clown hat

# In[7]:


plot_reward('smart_clown_hat')


# In[8]:


plot_reward_of_dropouts('smart_clown_hat')


# # EXP 14 - IDX 2506 - smart_clown_hat - 6 actions_1

# In[3]:


train_RL_model(14, 2506, 'smart_clown_hat', 6)


# In[4]:


visualise_avant_apres(14, 2506, 'smart_clown_hat', 6)


# In[6]:


idx = 2506
reward_name = 'smart_clown_hat'
nb_actions = 6

janus_env_2560  = Janus(idx, reward_name, nb_actions)
check_env(janus_env_2560)
janus_env_2560.reset()
RL_model = load_RL_model(14, idx, reward_name, nb_actions)


# In[7]:


print(f'Index de l observation {janus_env_2560.idx}')
current_position = janus_env_2560.revert_to_obs_space(janus_env_2560.full_x.iloc[janus_env_2560.idx].values.reshape(-1), janus_env_2560.full_x)
current_position = current_position.astype('float32')
current_y = janus_env_2560.ml_model.predict(janus_env_2560.convert_to_real_obs(current_position,
                                 janus_env_2560.full_x).values.reshape(1,-1)).reshape(-1)
print(f'init results:  {current_y} reward {janus_env_2560.reward(janus_env_2560.convert_to_real_obs(janus_env_2560.current_position, janus_env_2560.full_x).values.reshape(1,-1)):0.03f} action {janus_env_2560.last_action}')

for i in range(100):
    action, _ = RL_model.predict(janus_env_2560.current_position)
#     print(f'action {action}')
    obs, rewards, done, info = janus_env_2560.step(action)
#     janus_env.render()
    if done:
        print(f'done within {i+1} iterations')
        break
new_y = janus_env_2560.ml_model.predict(janus_env_2560.convert_to_real_obs(janus_env_2560.current_position,
                                     janus_env_2560.full_x).values.reshape(1,-1)).reshape(-1)

print(f'final results:  {new_y} reward {janus_env_2560.reward(janus_env_2560.convert_to_real_obs(janus_env_2560.current_position, janus_env_2560.full_x).values.reshape(1,-1)):0.03f} action {janus_env_2560.last_action}')


# In[10]:


janus_env_2560.reset()
janus_env_2560.last_action


# In[15]:


janus_env_2560.ml_model.predict(janus_env_2560.full_x)[2506]


# In[16]:


janus_env_2560.reward_smart_clown_hat([-1.12489282,  0.22756576])


# In[18]:


calculate_reward_dropouts(reward=janus_env_2560.reward_smart_clown_hat, reward_name='smart_clown_hat')


# In[19]:


reward_df = pd.read_csv('data/full_x_smart_clown_hat.csv', index_col=0)
reward_df


# In[20]:


reward_df.iloc[2506]


# In[21]:


plot_reward_of_dropouts('smart_clown_hat')


# # EXP 15 - IDX 2503 - smart_clown_hat

# In[22]:


train_RL_model(15, 2503, 'smart_clown_hat', 6)


# In[23]:


visualise_avant_apres(15,2503, 'smart_clown_hat', 6)


# In[ ]:




