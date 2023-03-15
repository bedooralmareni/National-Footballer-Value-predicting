# -*- coding: utf-8 -*-

# -- Sheet --

# # National Footballer Value predicting report 
# #  💵  ... 🏃‍♂️


# # Import Data


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


Players=pd.read_csv('players_20[1].csv')
Players['main_position']=Players['player_positions'].str.split(pat=',', n=-1, expand=True)[0]
Players.head(5)

# # Exploring the Dataset


Players.info()

Players.describe()

# ### Create a histogram of the main features


plt.style.use('fivethirtyeight')
ages=Players['age']
plt.hist(ages,bins=10,edgecolor='black')
plt.xlabel('age')

# ### Create a jointplot showing distributions of potential vs overall.
# some unusual behavior 


sns.jointplot(x='overall',y='potential',data=Players)

# ### Create a jointplot showing distributions of movement_sprint_speed vs movement_sprint_speed.
# Accelaration and SprintSpeed follow a proper linear relationship


sns.jointplot(x='movement_acceleration',y='movement_sprint_speed',data=Players)

# ### Create a jointplot showing distributions of Agility vs Stamina
# have linear relationship


sns.jointplot(x='movement_agility', y='power_stamina', data = Players)

# # Modeling
# - Simple Linear Regression.
#   
# - linear Regression with Data transformation.
#   
# - Grid Search & Random Forest regression technique.  


#Selected Feachers 

Players_new = Players[['short_name','value_eur','age', 'potential','shooting','passing','mentality_aggression']].dropna()
Players_new.head()

# Define X (feachers) and y (target)

X=Players_new.drop(Players_new[['short_name','value_eur']],axis=1).values 
y=Players_new['value_eur'].copy()
y=y.values/1000000

# Split data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train a simple linear regression
from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)

# Calculate evaluation metrics

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



predictions_1 = lin_reg.predict(X_test)
mse_1 = mean_squared_error(y_test,predictions_1)
rmse_1 = np.sqrt(mse_1)
mae_1 = mean_absolute_error(y_test,predictions_1)

mse_1, rmse_1, mae_1

# Score
import sklearn.metrics as sm

score_1 = round(sm.r2_score(y_test, predictions_1), 2)

print("R2 score =", score_1)


# Plot The Fitted Model
plt.figure(figsize=(19,10))
sns.regplot(predictions_1,y_test, marker="+", line_kws={'color':'darkred','alpha':1.0})

# # Data Processing 
# making improvement on previous results


# first, we remove goalkeepers cause they have statics vary from other players which effect the model accuracy 
# 
# secondarily, we abstract 42 relative features from 106 instead of 7 as before
# 
# thirdly, we will go through some data transformation functions that add a lot to improve our model


Players=Players[Players.main_position!='GK']
Skill_cols=['age', 'height_cm', 'weight_kg','potential',
       'international_reputation', 'weak_foot', 'skill_moves', 'pace',
       'shooting', 'passing', 'dribbling', 'defending', 'physic',
       'attacking_crossing', 'attacking_finishing',
       'attacking_heading_accuracy', 'attacking_short_passing',
       'attacking_volleys', 'skill_dribbling', 'skill_curve',
       'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
       'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
       'movement_reactions', 'movement_balance', 'power_shot_power',
       'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
       'mentality_aggression', 'mentality_interceptions',
       'mentality_positioning', 'mentality_vision', 'mentality_penalties',
       'mentality_composure', 'defending_marking', 'defending_standing_tackle',
       'defending_sliding_tackle']
print(len(Skill_cols))

# The object we edit X in transform is a dataframe that you access through the variable X. Unless you make a copy of this dataframe, the operations are inplace. Also, there is no need to return anything from the function


# transformaition feature
#class sklearn.base.BaseEstimator ,TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    #__init__ as explicit keyword arguments
    def __init__(self, Columns):
        self.Columns=Columns
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        New_X=X.copy()
        New_X=New_X[self.Columns].copy()
        return New_X

# Making a pipeline with custom transformer, SimpleImputer and StandardScaller
# 
# **pipeline :** Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on above data transform. 
# 
# **StandardScaller :** to make sure that it is standard normally distributed data becouse If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected. NaNs are treated as missing values: disregarded in fit, and maintained in transform.
# 
# **SimpleImputer :** for imputing / replacing numerical & categorical missing data using different strategies here we used median stratgy.


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pipeline=Pipeline([
    ('custom_tr', CustomTransformer(Skill_cols)),   #class from above
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler())
])
print(pipeline)

# 
# pipeline.fit_transform(Players) Fit the model and transform with the final estimator Returns:
# 
# transformed samples, (nd array of shape (n_samples, n_transformed_features))


X=pipeline.fit_transform(Players)
y=Players['value_eur'].copy()
y=y.values/1000000

# ### Data spliting
# 
# now we finished proceessing we start Data spliting into train & test


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# ### Linear Regression
# after we cleaned the data now, we apply linear regression again and see how that will affect our model accuracy


from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)

# Calculate evaluation metrics


from sklearn.metrics import mean_squared_error

predictions_2 = lin_reg.predict(X_test)
mse_2 = mean_squared_error(y_test,predictions_2)
rmse_2 = np.sqrt(mse_2)
mae_2 = mean_absolute_error(y_test,predictions_2)

mse_2, rmse_2, mae_2

# Score

score_2 = round(sm.r2_score(y_test, predictions_2), 2)

print("R2 score =", score_2)

# Plot The Fitted Model

plt.figure(figsize=(19,10))
sns.regplot(predictions_2,y_test, marker="+", line_kws={'color':'darkred','alpha':1.0})

# ### timeout for discussion
# as you may have noticed, the previous rmse was 4.8 and now become 4.0 that is a good thing but not good enough
# 
# let's explore a new improvement phase 😁


# ## Grid Search with Random Forest Regressor
# 
# The outcome of grid search is the optimal combination of one or more hyper parameters that gives the most optimal model complying to bias-variance tradeoff in other words finding the optimal set of parameters, through evaluation of all possible parameter combinations 
# 
# Random Forest has multiple decision trees as a base for learning models.
# we approach the Random Forest regression technique like any other machine learning technique. 


#it will take 57.6s run time 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid=[
    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8,10]},
    {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4,6]}
]
forest_reg=RandomForestRegressor()

grid_search=GridSearchCV(forest_reg, param_grid, cv=5,
                        scoring='neg_mean_squared_error',
                        return_train_score=True)

grid_search.fit(X_train,y_train)

# **Best parameters and score**
# 
# - best_params_ return: dict
# 
#         Parameter setting that gave the best results on the hold-out data.
# 
# - best_score_ return: float
# 
#         Mean cross-validated score of the best_estimator.


print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))

# **The most important features and skills**
# 
# - best_estimator
#   
#         Estimator that was chosen by the search,estimator which gave highest score(or smallest loss if specified).


feature_importances=grid_search.best_estimator_.feature_importances_
features=sorted(zip(feature_importances, Skill_cols),reverse=True)
features_sorted=np.array(features)

plt.pie(features_sorted[:,0], labels=features_sorted[:,1],radius=5,autopct='%1.1f%%')
plt.show()

# Set final model


final_model=grid_search.best_estimator_

# **Function for prediction of N top players of national teams**


# noinspection PyPep8Naming
def NationalTeamEstimator(nation,N=10):
    Players_National=Players[Players['nationality']==nation].copy()
    Players_National_prepared=pipeline.transform(Players_National)
    National_prediction=final_model.predict(Players_National_prepared)
    Players_National["value_predict"]=National_prediction
    Players_National=Players_National.sort_values(by='value_predict', ascending=False)
    Players_National["Model prediction"]=Players_National["value_predict"].round(2).astype(str)+" M Euro"
    Players_National["actual_value"]=(Players_National["value_eur"]/1e6).round(2).astype(str)+" M Euro"
    return (Players_National[['long_name','nationality','age','club','actual_value','Model prediction']].head(N))

NationalTeamEstimator('Argentina',N=20)

NationalTeamEstimator('Saudi Arabia',N=20)

NationalTeamEstimator('Morocco',N=20)

# Calculate evaluation metrics 


predictions_3 = final_model.predict(X_test)
mse_3 = mean_squared_error(y_test,predictions_3)
rmse_3 = np.sqrt(mse_3)
mae_3 = mean_absolute_error(y_test,predictions_3)

mse_3, rmse_3, mae_3

# Score

score_3 = round(sm.r2_score(y_test, predictions_3), 2)

print("R2 score =", score_3)

# Plot The Fitted Model
plt.figure(figsize=(19,10))
sns.regplot(predictions_3, y_test, marker="+", line_kws={'color':'darkred','alpha':1.0})

print("Simple Linear Regression RMSE =", rmse_1)
print("Simple Linear Regression Accurecy =", score_1)
print('')
print("linear Regression with Data transformation RMSE =", rmse_2)
print("linear Regression with Data transformation Accurecy =", score_2)
print('')
print("Grid Search & Random Forest regression RMSE =", rmse_3)
print("Grid Search & Random Forest regression Accurecy =", score_3) 

# # Results🏆🥅:
# 
# we managed to decrease RMSE from 5 to 4 at first, then ended up with 1.6, which is a big deal. Alhamdu llelah. 
# 
# you can check our report for more detail 
# 
# **team member :** Raneem Alomari,  Bedoor Alsulami, Alaa Yusuf




# -- Sheet 2 --

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


Players=pd.read_csv('players_20[1].csv')
Players['main_position']=Players['player_positions'].str.split(pat=',', n=-1, expand=True)[0]
Players.head(5)

Players.info()

Players.describe()

short_name = Players[['short_name']]
potential = Players[['potential']]
age = Players[['age']]
value_eur = Players[['value_eur']]
pace = Players[['pace']]
shooting = Players[['shooting']]
passing = Players[['passing']]
dribbling = Players[['dribbling']]
defending = Players[['defending']]


df_new = pd.concat([short_name,potential,value_eur,pace,shooting,passing,defending], axis = 1).dropna()
df_new.head()

Players_new = df_new[['short_name', 'potential','pace','shooting','passing','defending','value_eur']]
Players_new.head()

from sklearn.linear_model import LinearRegression

X = Players_new.iloc[:,[1,3]].values 
y=Players_new['value_eur'].copy()
y=y.values/1000000

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)

from sklearn.metrics import mean_squared_error

predictions=lin_reg.predict(X_test)
mse=mean_squared_error(y_test, predictions)
rmse=np.sqrt(mse)
mse, rmse

# ### improving


from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, Columns):
        self.Columns=Columns
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        New_X=X.copy()
        New_X=New_X[self.Columns].copy()
        return New_X

Players=Players[Players.main_position!='GK']
Skill_cols=['age', 'height_cm', 'weight_kg','potential',
       'international_reputation', 'weak_foot', 'skill_moves', 'pace',
       'shooting', 'passing', 'dribbling', 'defending', 'physic',
       'attacking_crossing', 'attacking_finishing',
       'attacking_heading_accuracy', 'attacking_short_passing',
       'attacking_volleys', 'skill_dribbling', 'skill_curve',
       'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
       'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
       'movement_reactions', 'movement_balance', 'power_shot_power',
       'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
       'mentality_aggression', 'mentality_interceptions',
       'mentality_positioning', 'mentality_vision', 'mentality_penalties',
       'mentality_composure', 'defending_marking', 'defending_standing_tackle',
       'defending_sliding_tackle']
print(len(Skill_cols))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pipeline=Pipeline([
    ('custom_tr', CustomTransformer(Skill_cols)),
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler())
])
print(pipeline)

X=pipeline.fit_transform(Players)
y=Players['value_eur'].copy()
y=y.values/1000000

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)

from sklearn.metrics import mean_squared_error

predictions=lin_reg.predict(X_test)
mse=mean_squared_error(y_test, predictions)
rmse=np.sqrt(mse)
mse, rmse





