#!/usr/bin/env python
# coding: utf-8

# ## pre load libs

# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from exp.janus_rl import *


# In[24]:


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


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_importance(importance,names,model_type,nb_features):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'][:nb_features], y=fi_df['feature_names'][:nb_features])
    #Add chart labels
    plt.title(model_type + ' - FEATURES IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


# # Identify features from our ml predict model linked to sub-quality

# In[5]:


sub_quality_range = (297, 324)

list_important_sub_quality = np.argsort(janus_env.ml_model.feature_importances_)[::-1]
list_important_sub_quality

list_important_features_real_column_names = [janus_env.full_x.columns[idx] for idx in list_important_sub_quality]
list_sub_quality_important_features_real_column_names = [qual for qual in list_important_features_real_column_names
                                                         if ( sub_quality_range[0] <= int(qual.split('_')[1]) <= sub_quality_range[1])]


# In[6]:


list_sub_quality_important_features_real_column_names[:7]


# I will focus arbitrarely on these 7 first features

# # train a prediction model for these features

# In[8]:





sub_quality_col = list_sub_quality_important_features_real_column_names[:7]


y_df = janus_env.full_x[sub_quality_col].copy()
x_df = janus_env.full_x.drop(columns=sub_quality_col).copy()

print('features shape: {}, \nsub quality targets shape: {}'.format(x_df.shape, y_df.shape))

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.1, random_state=14)
print('\nLength of train is {}, test is {}'.format(len(x_train), len(x_test)))


# In[9]:


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


# In[10]:


## ---- save/load the model to disk

## Random forest
filename = 'data/models/janus_stam_RF.pkl'  # janus_LinearReg, janus_RF

# pickle.dump(ml_model, open(filename, 'wb'))

# # load the model from disk
ml_model = pickle.load(open(filename, 'rb'))
print(f'R squared: {ml_model.score(x_test, y_test.values):0.04f}')


# Check if we don't have too much sub quality inputs in feature importance

# In[11]:


list_important_features = np.argsort(ml_model.feature_importances_)
list_important_features


# In[12]:


[janus_env.full_x.columns[idx] for idx in list_important_features]


# In[ ]:


import shap
explainer = shap.TreeExplainer(ml_model)
shap_values = explainer.shap_values(x_test)


# In[35]:


shap.summary_plot(shap_values, x_test)


# In[17]:


plot_feature_importance(ml_model.feature_importances_,x_df.columns,'RANDOM FOREST', 15)


# # Retrain a model without any sub-quality data

# In[23]:


sub_quality_range = (297, 324)
sub_quality_col = [qual for qual in janus_env.full_x.columns
                                                         if ( sub_quality_range[0] <= int(qual.split('_')[1]) <= sub_quality_range[1])]
sub_quality_col


# In[25]:


y_df = janus_env.full_x[sub_quality_col].copy()
x_df = janus_env.full_x.drop(columns=sub_quality_col).copy()

print('features shape: {}, \nsub quality targets shape: {}'.format(x_df.shape, y_df.shape))

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.1, random_state=14)
print('\nLength of train is {}, test is {}'.format(len(x_train), len(x_test)))


# In[26]:


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


# In[27]:


## ---- save/load the model to disk

## Random forest
filename = 'data/models/janus_stam_RF.pkl'  # janus_LinearReg, janus_RF

# pickle.dump(ml_model, open(filename, 'wb'))

# # load the model from disk
ml_model = pickle.load(open(filename, 'rb'))
print(f'R squared: {ml_model.score(x_test, y_test.values):0.04f}')


# In[28]:


plot_feature_importance(ml_model.feature_importances_,x_df.columns,'RANDOM FOREST', 15)


# In[31]:


janus_env = Janus(nbr_actions=6)

janus_env.full_x.columns[janus_env.list_important_actions]


# In[ ]:




