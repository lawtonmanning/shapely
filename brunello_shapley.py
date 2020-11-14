import pandas as pd
%load_ext autoreload
%autoreload 2
import os
import sys
import time
import numpy as np
from Shapley import ShapNN
from DShap import DShap
import matplotlib.pyplot as plt
import sklearn
from shap_utils import *
from scipy import stats
%matplotlib inline
MEM_DIR = './'

train_df = pd.read_csv("Data/brunello_train_shap.csv")
test_df = pd.read_csv("Data/brunello_test_shap.csv")

print(train_df.shape)
print(test_df.shape)

X, y = np.array(train_df.iloc[:,1:-1]), np.array(train_df.iloc[:,-1])
X_test, y_test = np.array(test_df.iloc[:,1:-1]), np.array(test_df.iloc[:,-1])
y = y.astype(int)
y_test = y_test.astype(int)

model = 'logistic'
problem = 'classification'
num_test = 1000
directory = './temp'
dshap = DShap(X, y, X_test, y_test, num_test,
              sources=None,
              sample_weight=None,
              model_family=model,
              metric='accuracy',
              overwrite=True,
              directory=directory, seed=0)
dshap.run(100, 0.1, g_run=False)

dshap.merge_results()

tmc_values = dshap.values_tmc
loo_values = dshap.vals_loo
print(tmc_values)
train_df["shapley vals"] = tmc_values
#train_df.to_csv("brunello_train_with_shapley.csv",index = False)


print(stats.describe(tmc_values))
print(stats.describe(loo_values))

fig1 = dshap.performance_plots([dshap.values_tmc, dshap.vals_loo], num_plot_markers=20,
                       sources=dshap.sources, order = "d")
#fig1.savefig("Logistic_accuracy_exp1.pdf")

fig2 = dshap.performance_plots([dshap.values_tmc, dshap.vals_loo], num_plot_markers=20,
                       sources=dshap.sources,order = "a")
#fig2.savefig("Logistic_accuracy_exp2.pdf")

#plt.figure()

#plt.style.use('ggplot')
#plt.hist(tmc_values, bins = 10)
#plt.show
#plt.savefig("logistic_accuracy_shap_hist.pdf")
