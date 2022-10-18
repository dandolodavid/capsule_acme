import json
import os
import pandas as pd
import numpy as np
import time
import warnings
import shap
import matplotlib.pyplot as plt
from ACME.ACME import ACME
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

def save_images(shap_val,acme,name):

    shap.summary_plot(shap_val, dataframe.drop(columns='target'),plot_size=(15,10), show=False)
    plt.savefig('../results/acme_vs_shap/shap_'+name+'.pdf', format='pdf')
    shap.summary_plot(shap_val, dataframe.drop(columns='target'), show=False, plot_size=(15,10), plot_type='bar')
    plt.savefig('../results/acme_vs_shap/SHAP_bar_'+name+'.pdf', format='pdf')

    fig = acme.summary_plot()
    fig.update_layout(height=650).write_image('../results/acme_vs_shap/ACME_'+name+'.pdf')
    fig = acme.bar_plot()
    fig.update_layout(height=650).write_image('../results/acme_vs_shap/ACME_bar_'+name+'.pdf')

os.mkdir('../results/acme_vs_shap')

# Full Boston dataset
from sklearn import datasets
boston = datasets.load_boston()
X = boston.data
y = boston.target
        
dataframe = pd.DataFrame(X, columns=boston.feature_names)
dataframe['target'] = y
features = dataframe.drop(columns={'target'}).columns


# Train different model
print('Model train...')
models = {}
models['linear_regression'] = LinearRegression().fit(X,y)
models['random_forest_regressor'] = RandomForestRegressor().fit(X,y)
models['cat_boost_regressor'] = CatBoostRegressor().fit(X,y)
models['svr'] = SVR().fit(X,y) 
models['xgboost'] = xgb.XGBRegressor().fit(X,y)

# MSE for models

print('Model MSE...')
from sklearn.metrics import mean_squared_error
for model in models.keys():
    pred = models[model].predict(X)
    mse = mean_squared_error(y, pred)
    print(model +' '+ 'MSE: ' + str(mse))

# Compare ACME and SHAP results
print('Starting explainability...')
time_start = {}
time_elapsed = {}

# XGBoost
## ACME
time_start = time.time()
acme_xg = ACME(models['xgboost'],'target')
acme_xg = acme_xg.fit(dataframe, robust=True)
time_elapsed['ACME_XG'] = (time.time() - time_start)
## SHAP
time_start = time.time()
shap_xg = shap.KernelExplainer(models['xgboost'].predict,X)
shap_xg_values = shap_xg.shap_values(X)
time_elapsed['SHAP_XG'] = (time.time() - time_start)

save_images(shap_xg_values,acme_xg,'XG')

# Linear Regression
## ACME
time_start = time.time()
acme_lr = ACME(models['linear_regression'],'target')
acme_lr = acme_lr.fit(dataframe, robust=True)
time_elapsed['ACME_LR'] = (time.time() - time_start)

## SHAP
time_start = time.time()
shap_lr = shap.KernelExplainer(models['linear_regression'].predict,X)
shap_lr_values = shap_lr.shap_values(X)
time_elapsed['SHAP_LR'] = (time.time() - time_start)

save_images(shap_lr_values,acme_lr,'LR')

# Random Forest
## ACME
time_start = time.time()
acme_rf = ACME(models['random_forest_regressor'],'target')
acme_rf = acme_rf.fit(dataframe, robust=True)
time_elapsed['ACME_RF'] = (time.time() - time_start)

## SHAP
time_start = time.time()
shap_rf = shap.KernelExplainer(models['random_forest_regressor'].predict, X)
shap_rf_values = shap_rf.shap_values(X)
time_elapsed['SHAP_RF'] = (time.time() - time_start)

save_images(shap_rf_values,acme_rf,'RF')

# Cat_boost_regressor
## ACME
time_start = time.time()
acme_ct = ACME(models['cat_boost_regressor'],'target')
acme_ct = acme_ct.fit(dataframe,robust=True)
time_elapsed['ACME_CT'] = (time.time() - time_start)

## SHAP
time_start = time.time()
shap_ct = shap.KernelExplainer(models['cat_boost_regressor'].predict, X)
shap_ct_values = shap_ct.shap_values(X)
time_elapsed['SHAP_CT'] = (time.time() - time_start)

save_images(shap_xg_values,acme_xg,'XG')

# SVR
## ACME
time_start = time.time()
acme_svr = ACME(models['svr'],'target')
acme_svr = acme_svr.fit(dataframe, robust=True)
time_elapsed['ACME_SVR'] = (time.time() - time_start)

## SHAP
time_start = time.time()
shap_svr = shap.KernelExplainer(models['svr'].predict,  X)
shap_svr_values = shap_svr.shap_values(X)
time_elapsed['SHAP_SVR'] = (time.time() - time_start)

save_images(shap_svr_values,acme_svr,'XG')

# RESULTS
with open('../results/acme_vs_shap/acme_vs_shap_time_elapsed.txt', 'w') as time_elapsed_file:
    time_elapsed_file.write(json.dumps(time_elapsed))




