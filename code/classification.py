import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from ACME.ACME import ACME 
from plotly.io import write_image
import matplotlib.pyplot as plt

os.mkdir('../results/classification')

glass_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data',index_col=[0])
glass_data.columns = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type']
features = glass_data.drop(columns='Type').columns.to_list()

# Classifier
rfc = RandomForestClassifier(verbose=False).fit( glass_data[features], glass_data['Type'].values )

# ACME
acme_rf = ACME(rfc, 'Type',K=50, task = 'class')
acme_rf = acme_rf.fit(glass_data,robust = True, label_class=None)

# ACME class 1
acme_class =  acme_rf.fit(glass_data, robust = True, label_class=1)
fig = acme_class.summary_plot()
fig.update_layout(height=650).write_image('../results/classification/ACME_class.pdf')
fig = acme_class.bar_plot()
fig.update_layout(height=650).write_image('../results/classification/ACME_bar_class.pdf')

# ACME Local
local_acme = acme_class.fit_local(glass_data,local=100, label_class=1)
fig = local_acme.summary_plot(local=True)
fig.write_image('../results/classification/ACME_bar_local.pdf')


