#!/usr/bin/env python
# coding: utf-8

# # load janus_rl library

# In[7]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from exp.janus_rl import *


# # experiments

# ## IDX 7547

# In[4]:


plot_reward_of_dropouts('clown_hat')


# ### IDX 7547 - EXP 12 - clown hat

# In[5]:


train_RL_model(12, 7547, 'clown_hat')


# In[8]:


visualise_avant_apres(12, 7547, 'clown_hat')


# ### IDX 7547 - EXP 12 - smart archery

# In[14]:


train_RL_model(12, 7547, 'smart_archery')


# In[15]:


visualise_avant_apres(12, 7547, 'smart_archery')


# ### IDX 7547 - EXP 12 - archery

# In[16]:


train_RL_model(12, 7547, 'archery')


# In[17]:


visualise_avant_apres(12, 7547, 'archery')


# # plot all dropout points on this target square

# In[61]:


reward_name = 'clown_hat'
reward_df=pd.read_csv('./data/full_x_'+reward_name+'.csv', index_col=0)



fig = go.Figure()

fig.update_xaxes(title_text='target_0', range=[inf_limx-3, sup_limx+3])
fig.update_yaxes(title_text='target_5', range=[inf_limy-3, sup_limy+3])

fig.add_trace(go.Scatter(x=[inf_x,inf_x,sup_x,sup_x, inf_x, None, vav_x], y=[inf_y,sup_y,sup_y, inf_y, inf_y, None, vav_y], fill="toself", name='TI, TS, VAV'))
# for row in reward_df.iterrows():
fig.add_trace(go.Scatter(
    mode='markers',
    x=reward_df['target_0'].values,
    y=reward_df['target_5'].values,
    marker=dict(
        color='rgba(255, 0, 0, 0.1)',
        size=5,
    ),
    hovertemplate='target_0 <b>%{x:0.02f}</b> <br>target_5 <b>%{y:0.02f}</b> <br>%{text} ',
    text = ['Tombee {}'.format(i) for i in range(len(reward_df))],
    
    showlegend=True,
    name='reward_df'
))
        
        
fig.update_layout({'showlegend': True, 'title':'Positions qualité distribution complète'})
fig.show()


# In[84]:


import plotly.graph_objects as go

frame=[]
nbr_point = 100

for i in range(len(reward_df) // nbr_point):
    frame.append(go.Frame(data=[go.Scatter(mode='markers',
    x=reward_df[nbr_point*i: nbr_point*(i+1)]['target_0'].values,
    y=reward_df[nbr_point*i: nbr_point*(i+1)]['target_5'].values,
    marker=dict(
        color='rgba(255, 0, 0, 0.1)',
        size=5,
    ))]))
    
fig = go.Figure(
    data=[go.Scatter(    mode='markers',
    x=reward_df[:nbr_point]['target_0'].values,
    y=reward_df[:nbr_point]['target_5'].values,
    marker=dict(
        color='rgba(255, 0, 0, 0.1)',
        size=5,
    )
                    )],
    layout=go.Layout(
        xaxis=dict(range=[-6, 6], autorange=False),
        yaxis=dict(range=[-6, 6], autorange=False),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 100, 
                                                                        "redraw": False},
                                                              "fromcurrent": True, 
                                                              "transition": {"duration": 20}}])])]
    ),
    frames=frame
    )

fig.show()


# In[ ]:




