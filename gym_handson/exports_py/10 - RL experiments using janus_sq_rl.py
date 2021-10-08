#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from exp.janus_sq_rl import *


# # reward smart clown hat

# In[2]:


plot_reward('smart_clown_hat')


# In[3]:


plot_reward_of_dropouts('smart_clown_hat')


# # EXP 16 - IDX 2506 - smart_clown_hat - 6 actions_1

# In[4]:


train_RL_model(16, 2506, 'smart_clown_hat', 6)


# In[5]:


visualise_avant_apres(16, 2506, 'smart_clown_hat', 6)


# # EXP 17 - IDX 2503 - smart_clown_hat

# In[6]:


train_RL_model(17, 2503, 'smart_clown_hat', 6)


# In[7]:


visualise_avant_apres(17,2503, 'smart_clown_hat', 6)


# # EXP 18-19-20 IDX 47 2230 11926

# In[10]:


idx = [47, 2230, 11926]
for exp, index in enumerate(idx):
    print(exp+18, index)
    train_RL_model(exp+18, index, 'smart_clown_hat', 6)


# In[11]:


idx = [47, 2230, 11926]
for exp, index in enumerate(idx):
    print(exp+18, index)
    visualise_avant_apres(exp+18, index, 'smart_clown_hat', 6)


# In[ ]:




