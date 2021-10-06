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

# In[5]:


train_RL_model(14, 2506, 'smart_clown_hat', 6)


# In[6]:


visualise_avant_apres(14, 2506, 'smart_clown_hat', 6)


# In[ ]:




