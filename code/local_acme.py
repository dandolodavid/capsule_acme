import os
import pandas as pd
import numpy as np
from ACME.ACME import ACME 
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target        
dataset = pd.DataFrame(X, columns=boston.feature_names)
dataset['target'] = y

rf = RandomForestRegressor(n_estimators=200)
rf.fit(dataset.drop(columns='target'),y)

os.mkdir('../results/local_acme')
acme_rf = ACME(rf,'target',qualitative_features=['CHAS'], K=50)
acme_rf = acme_rf.fit(dataset)
acme_local = acme_rf.fit_local(dataset, local=100)
acme_local.local_table().head()
fig_local = acme_local.summary_plot(local=True)
fig_local.write_image('../results/local_acme/fig_local.png')





