#!/usr/bin/env python
# coding: utf-8

# # load janus_rl library

# In[7]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from exp.janus_rl import *


# # experiments

# ## reward clown_hat

# In[4]:


plot_reward_of_dropouts('clown_hat')


# ### IDX 7547 - EXP 12 - clown hat

# In[5]:


train_RL_model(12, 7547, 'clown_hat')


# In[8]:


visualise_avant_apres(12, 7547, 'clown_hat')


# In[ ]:




