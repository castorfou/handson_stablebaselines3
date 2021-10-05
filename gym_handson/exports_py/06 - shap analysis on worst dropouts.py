#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from exp.janus_rl import *


# In[2]:


import shap
print('shap version: ',shap.__version__)


# # calculate shap values

# In[4]:


import shap
explainer = shap.TreeExplainer(janus_env.ml_model)


# In[5]:


shap_values = explainer.shap_values(janus_env.full_x)


# In[6]:


shap.summary_plot(shap_values, janus_env.full_x)


# # multi regression case

# In[8]:


idx = 100
shap_value_single = explainer.shap_values(X = janus_env.full_x.iloc[idx:idx+1,:])


# In[9]:


list_of_labels = janus_env.y_df.columns.to_list()


# In[12]:


# Create a list of tuples so that the index of the label is what is returned
tuple_of_labels = list(zip(list_of_labels, range(len(list_of_labels))))

current_label = 'target_0'


# In[13]:


# print the JS visualization code to the notebook
shap.initjs()

print(f'Current label Shown: {list_of_labels[0]}')

shap.force_plot(base_value = explainer.expected_value[0],
                shap_values = shap_value_single[0],
                features = janus_env.full_x.iloc[idx:idx+1,:]
                )


# In[14]:


shap.summary_plot(shap_values = shap_value_single[0],
                  features = janus_env.full_x.iloc[idx:idx+1,:]
                  )


# ## get worst observations

# In[3]:


reward_name='clown_hat'
reward_df=pd.read_csv('./data/full_x_'+reward_name+'.csv', index_col=0)

reward_df.sort_values(by='reward_clown_hat')
worst_dropouts = reward_df[reward_df.reward_clown_hat<=-10]

worst_dropouts


# ## apply shap

# In[22]:


shap_values_worst = explainer.shap_values(X = janus_env.full_x.iloc[worst_dropouts.index,:])


# In[23]:


shap.summary_plot(shap_values = shap_values_worst[0],
                  features = janus_env.full_x.iloc[worst_dropouts.index,:]
                  )


# In[24]:


shap.summary_plot(shap_values = shap_values_worst[1],
                  features = janus_env.full_x.iloc[worst_dropouts.index,:]
                  )


# In[25]:


shap.initjs()

shap.force_plot(base_value = explainer.expected_value[0],
                shap_values = shap_values_worst[0],
                features = janus_env.full_x.iloc[worst_dropouts.index,:]
                )


# ## get highest effects from controllable parameters

# From target_0:
# * data_54
# * data_41
# * data_68
# * data_51
# * data_6
# 
# From target_5:
# * data_6
# * data_55
# * data_54
# * data_68

# Let's keep these 6 features in gym
# 
# * data_6
# * data_41
# * data_51
# * data_54
# * data_55
# * data_68
# 
# it matches with `new_list = np.argsort(janus_env.ml_model.feature_importances_[:94])[::-1][:6]`

# # Train a model with 6 controlable parameters

# In[5]:


worst_dropouts.sample()


# In[9]:


worst_dropouts.sample().index.values[0]


# In[10]:


train_RL_model(13, worst_dropouts.sample().index.values[0], 'clown_hat', nb_actions=6)


# In[3]:


visualise_avant_apres(13, 2506, 'clown_hat', 6)


# In[ ]:




