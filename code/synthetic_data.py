import sys
import os
import pandas as pd
import numpy as np
import time
import warnings
import shap
from ACME.ACME import ACME 
from sklearn.metrics import ndcg_score
from sklearn.linear_model import LinearRegression
from plotly.io import write_image
import matplotlib.pyplot as plt
import json


os.mkdir('../results/synt_data')

warnings.filterwarnings("ignore")
time_elapsed = {}
ndcg_dict = {}

#Model 1
## Definitions
mu = [10, 10, 10, 10, 10, 10, 10, 10]
covariance=[[10,0,0,0,0,0,0,0],
           [0,10,0,0,0,0,0,0],
           [0,0,10,0,0,0,0,0],
           [0,0,0,10,0,0,0,0],
           [0,0,0,0,10,0,0,0],
           [0,0,0,0,0,10,0,0],
           [0,0,0,0,0,0,10,0],
           [0,0,0,0,0,0,0,10]]

X = np.random.multivariate_normal(mu, covariance, size=200)
beta = np.array([10, 20, -10, 0.3, 1, 0, 0, -0.5 ])
df_1 = pd.DataFrame(X)
df_1.columns = ['x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8']
df_1['target']=  beta.dot(X.T) + np.random.normal(0,10,200)

## Model build
reg = LinearRegression().fit(df_1[['x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8']].values, df_1['target'].values)

## Shap interpretability
time_start = time.time()
shap_mod1 = shap.KernelExplainer(reg.predict, df_1.drop(columns='target'))
shap_values = shap_mod1.shap_values( df_1.drop(columns='target') )
time_elapsed['model_1'] = (time.time() - time_start)

## AcME interpretabilty
time_start = time.time()
acme_mod1 = ACME(reg, 'target', K=50)
acme_mod1 = acme_mod1.fit(df_1, robust=True)
time_elapsed['model_1_ashap'] = (time.time() - time_start)

# save plot
shap.summary_plot(shap_values, df_1.drop(columns='target'),plot_size=(15,10), show=False)
plt.savefig('../results/synt_data/shap_mod1.pdf', format='pdf')

fig = acme_mod1.summary_plot()
fig.write_image("../results/synt_data/acme_mod1.png")

#save score
score = acme_mod1.feature_importance().loc[['x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8']].reset_index()
score['true'] = np.abs(beta)
score['Importance'] = score['Importance']/sum(score['Importance'])
ndcg_dict['model_1'] = ndcg_score([score['true'].tolist()],[score['Importance'].tolist()])


#Model 2
mu = [100, 10, 10, 10, 100, 10, 10, 100]
covariance=[[100,0,0,0,0,0,0,0],
           [0,10,0,0,0,0,0,0],
           [0,0,10,0,0,0,0,0],
           [0,0,0,10,0,0,0,0],
           [0,0,0,0,100,0,0,0],
           [0,0,0,0,0,10,0,0],
           [0,0,0,0,0,0,10,0],
           [0,0,0,0,0,0,0,100]]

X = np.random.multivariate_normal(mu, covariance, size= 200)
features=['x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8']
beta = np.array([10, 20, -10, 0.3, 1, 0, 0, -0.5])
df_2 = pd.DataFrame(X)
df_2.columns = ['x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8']
df_2['target']=  beta.dot(df_2.T.to_numpy()) + np.random.normal(0,10,200)

## Model build
reg = LinearRegression().fit(df_2[features].values, df_2['target'].values)

## Shap interpretability
time_start = time.time()
shap_mod2 = shap.KernelExplainer(reg.predict, df_2[features] )
shap_values2 = shap_mod2.shap_values( df_2[features] )
 
## Acme interpretability
time_elapsed['model_2'] = (time.time() - time_start)
time_start = time.time()
acme_mod2 = ACME(reg, 'target', K=50)
acme_mod2 = acme_mod2.fit(df_2[features+['target']], robust = True)
time_elapsed['model_2_ashap'] = (time.time() - time_start)

# save plot
shap.summary_plot(shap_values2, df_2[features],plot_size=(15,10), show=False)
plt.savefig('../results/synt_data/shap_mod2.pdf', format='pdf')

fig = acme_mod2.summary_plot()
fig.write_image("../results/synt_data/acme_mod2.png")

#save score
score = acme_mod2.feature_importance().loc[['x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8']].reset_index()
score['true'] = np.abs(beta*mu)
score['Importance'] = score['Importance']/sum(score['Importance'])
ndcg_dict['model_2'] = ndcg_score([score['true'].tolist()],[score['Importance'].tolist()])


# Model 3

## Definitions
features=['x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8']
beta = np.array([10, 20, -10, 0, 0, 0, 0, -0.5, 
                 5, 5, 2, 2, 10,10])
df_3 = pd.DataFrame(X)
df_3.columns = ['x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8']
df_3['interaction_1_2'] = df_3['x_1'].values * df_3['x_2'].values/10
df_3['interaction_1_7'] = df_3['x_1'].values * df_3['x_2'].values/10
df_3['interaction_1_8'] = df_3['x_1'].values * df_3['x_4'].values/10
df_3['interaction_2_6'] = df_3['x_2'].values * df_3['x_6'].values/10
df_3['interaction_7_8'] = df_3['x_7'].values * df_3['x_8'].values/10
df_3['interaction_6_8'] = df_3['x_6'].values * df_3['x_8'].values/10
df_3['target']=  beta.dot(df_3.T.to_numpy()) + np.random.normal(0,10,200)

## Model build
reg = LinearRegression().fit(df_3[features].values, df_3['target'].values)

## Shap interpretability
time_start = time.time()
shap_mod3 = shap.KernelExplainer(reg.predict, df_2[features] )
shap_values3 = shap_mod3.shap_values( df_3[features] )
time_elapsed['model_3'] = (time.time() - time_start)

## AcME interpretability
time_start = time.time()
acme_mod3 = ACME(reg, 'target', K=50)
acme_mod3 = acme_mod3.fit(df_3[features+['target']], robust=True)
time_elapsed['model_3_ashap'] = (time.time() - time_start)

# save plot
shap.summary_plot(shap_values3, df_3[features],plot_size=(15,10), show=False)
plt.savefig('../results/synt_data/shap_mod3.pdf', format='pdf')
fig = acme_mod3.summary_plot()
fig.write_image("../results/synt_data/acme_mod3.png")

print('Synthetic dataset complete')
print(time_elapsed)
print(ndcg_dict)


with open('../results/synt_data/synthetic_data_time_elapsed.txt', 'w') as time_elapsed_file:
     time_elapsed_file.write(json.dumps(time_elapsed))

with open('../results/synt_data/synthetic_data_ndgc.txt', 'w') as ndcg_dict_file:
     ndcg_dict_file.write(json.dumps(ndcg_dict))