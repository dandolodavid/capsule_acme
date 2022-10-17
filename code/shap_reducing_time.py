# %%
import json
import os
import pandas as pd
import numpy as np
import time
import warnings
import shap
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings("ignore")

# Data
from sklearn import datasets
boston = datasets.load_boston()
X = boston.data
y = boston.target
        
dataframe = pd.DataFrame(X, columns=boston.feature_names)
dataframe['target'] = y
features = dataframe.drop(columns={'target'}).columns

# Model
rf = RandomForestRegressor().fit(X,y)
n_sample_list = [5,10,20,100]

# N coalitions
time_elapsed = {}
os.mkdir('../results/shap_ncoalitions/')
for ns in n_sample_list:
    shap_ke = shap.KernelExplainer(rf.predict, X)
    start = time.time()
    shap_ke_val = shap_ke.shap_values(X, nsamples=100)
    time_elapsed[str(ns)] = time.time()-start
    shap.summary_plot(shap_ke_val, X,plot_size=(15,10), show=False)
    plt.savefig('../results/shap_ncoalitions/shap_'+name+'.pdf', format='pdf')

with open('../results/shap_ncoalitions/shap_ncoalitions.txt', 'w') as time_elapsed_file:
     time_elapsed_file.write(json.dumps(time_elapsed))


# Subsample
os.mkdir('../results/shap_nsamples/')
time_elapsed = {}
for ns in n_sample_list:
    shap_ke = shap.KernelExplainer(rf.predict, shap.sample(X,10))
    start = time.time()
    shap_ke_val = shap_ke.shap_values(shap.sample(X,10))
    time_elapsed[str(ns)] = time.time() - start
    shap.summary_plot(shap_ke_val, X, plot_size=(15,10), show=False)
    plt.savefig('../results/shap_nsample/shap_'+name+'.pdf', format='pdf')


with open('../results/shap_nsample/shap_nsample.txt', 'w') as time_elapsed_file:
     time_elapsed_file.write(json.dumps(time_elapsed))

