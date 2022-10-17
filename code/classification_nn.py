import os
import pandas as pd
import numpy as np
from ACME.ACME import ACME 
import time
import shap
import warnings
import json
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def save_images(shap_val,acme,name):

    shap.summary_plot(shap_val, glass_data.drop(columns='Type'),plot_size=(15,10), show=False)
    plt.savefig('../results/classification_nn/shap_'+name+'.pdf', format='pdf')
    shap.summary_plot(shap_val, glass_data.drop(columns='Type'), show=False, plot_size=(15,10), plot_type='bar')
    plt.savefig('../results/classification_nn/SHAP_bar_'+name+'.pdf', format='pdf')

    fig = acme.summary_plot()
    fig.update_layout(height=650).write_image('../results/classification_nn/ACME_'+name+'.pdf')
    fig = acme.bar_plot()
    fig.update_layout(height=650).write_image('../results/classification_nn/ACME_bar_'+name+'.pdf')


# data prep
glass_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data',index_col=[0])
glass_data.columns = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type']
features = glass_data.drop(columns='Type').columns.to_list()
X = glass_data.drop('Type',axis=1)
y = glass_data['Type']
from sklearn.model_selection import train_test_split
train, test = train_test_split(glass_data, test_size=0.3, stratify=glass_data['Type'].values, random_state=1234)

# Mlp train
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(9,15,15,6),max_iter=1000, random_state=678)
mlp.fit(X,y)
from sklearn.metrics import accuracy_score
accuracy_score(test['Type'].values, mlp.predict(test[features].values), normalize=True)

time_elapsed = {}
## ACME
time_start = time.time()
acme_nn = ACME(mlp, 'Type',K=20, task = 'c')
acme_nn = acme_nn.fit(glass_data)
time_elapsed['ACME'] = time.time() - time_start

# SHAP
time_start = time.time()
shap_nn = shap.KernelExplainer(mlp.predict_proba,X)
shap_nn_values = shap_nn.shap_values(X)
time_elapsed['SHAP'] = time.time() - time_start

os.mkdir('../results/classification_nn')
save_images(shap_nn_values, acme_nn, 'MLP')

with open('../results/classification_nn/classification_nn.txt', 'w') as time_elapsed_file:
    time_elapsed_file.write(json.dumps(time_elapsed))