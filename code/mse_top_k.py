import pandas as pd
import os
import numpy as np
import time
import warnings
import shap
from ACME.ACME import ACME
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import json

from sklearn import datasets
boston = datasets.load_boston()
X = boston.data
y = boston.target
dataframe = pd.DataFrame(X, columns=boston.feature_names)
dataframe['target'] = y

features = dataframe.drop(columns='target').columns.tolist()
X = dataframe[features]
y = dataframe['target']

models = {}
models['linear_regression'] = LinearRegression().fit(X,y)
models['random_forest_regressor'] = RandomForestRegressor().fit(X,y)
models['cat_boost_regressor'] = CatBoostRegressor(verbose=False).fit(X,y)
models['svr'] = SVR().fit(X,y) 
models['xgboost'] = xgb.XGBRegressor().fit(X,y)

mse_full = {}
for model in models.keys():
    mse_full[model] = mean_squared_error(models[model].predict(X),y)

acme = {}
for model in models.keys():
    acme[model] = ACME(models[model],'target')
    acme[model] = acme[model].fit(dataframe, robust=True)

k = 5
top_k = {}
last_k = {}
for model in models.keys():
    top_k[model] = acme[model].feature_importance().index.tolist()[0:k]
    last_k[model] = acme[model].feature_importance().index.tolist()[-k:]

models_top_k = {}
models_top_k['linear_regression'] = LinearRegression().fit(dataframe[top_k['linear_regression']],y)
models_top_k['random_forest_regressor'] = RandomForestRegressor().fit(dataframe[top_k['random_forest_regressor']],y)
models_top_k['cat_boost_regressor'] = CatBoostRegressor(verbose=False).fit(dataframe[top_k['cat_boost_regressor']],y)
models_top_k['svr'] = SVR().fit(dataframe[top_k['svr']],y)
models_top_k['xgboost'] = xgb.XGBRegressor().fit(dataframe[top_k['xgboost']],y)

mse_top_k = {}
for model in models.keys():
    mse_top_k[model] = mean_squared_error(models_top_k[model].predict(dataframe[top_k[model]]),y)

models_last_k = {}
models_last_k['linear_regression'] = LinearRegression().fit(dataframe.drop(columns = top_k['linear_regression'] + ['target']),y)
models_last_k['random_forest_regressor'] = RandomForestRegressor().fit(dataframe.drop(columns = top_k['random_forest_regressor']+ ['target']),y)
models_last_k['cat_boost_regressor'] = CatBoostRegressor(verbose=False).fit(dataframe.drop(columns = top_k['cat_boost_regressor']+ ['target']),y)
models_last_k['svr'] = SVR().fit(dataframe.drop(columns = top_k['svr']+ ['target']),y)
models_last_k['xgboost'] = xgb.XGBRegressor().fit(dataframe.drop(columns = top_k['xgboost']+ ['target']),y)

mse_last_k = {}
for model in models.keys():
    mse_last_k[model] = mean_squared_error(models_last_k[model].predict(dataframe.drop(columns = top_k[model]+ ['target'])),y)


os.mkdir('../results/mse_top_k')

with open('../results/mse_top_k/mse_full.txt', 'w') as mse_full_file:
     mse_full_file.write(json.dumps(mse_full))

with open('../results/mse_top_k/mse_last_k.txt', 'w') as mse_last_k_file:
     mse_last_k_file.write(json.dumps(mse_last_k))

with open('../results/mse_top_k/mse_top_k.txt', 'w') as mse_top_k_file:
    mse_top_k_file.write(json.dumps(mse_top_k))



