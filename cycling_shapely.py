import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from os import listdir
from os.path import join, isdir

from jenkspy import jenks_breaks

from math import inf

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'axes.unicode_minus': False,
})

from DShap import DShap

import pickle as pck

from scipy import stats



data_dir = join('data', 'cycling')



# Load data
train_data = pd.read_pickle(join(data_dir,'train','train_data.pkl'))
test_data = pd.read_pickle(join(data_dir,'test','test_data.pkl'))

train_labels = np.load(join(data_dir,'train','train_labels.npy'))
test_labels = np.load(join(data_dir,'test','test_labels.npy'))

value_data = pd.read_pickle(join(data_dir, 'train', 'value_data.pkl'))
eval_data = pd.read_pickle(join(data_dir, 'train', 'eval_data.pkl'))

value_labels = np.load(join(data_dir, 'train', 'value_labels.npy'))
eval_labels = np.load(join(data_dir, 'train', 'eval_labels.npy'))


# compute Shapely and LOOV using models and metrics
model_families = ['NB', 'LinearSVC']
metrics = ['accuracy', 'auc']

for model_family in model_families:
    for metric in metrics:
        if model_family == 'LinearSVC' and metric == 'auc':
            continue
        for seed in range(1,4):
            dshap = DShap(value_data.to_numpy(), value_labels, eval_data.to_numpy(), eval_labels, 1162, 
                          directory=join('output','cycling',model_family,metric),
                          model_family=model_family, metric=metric, seed=seed, n_neighbors=3)
            dshap.run(100, 0.1)
        dshap.merge_results()
        
        fig = dshap.performance_plots([dshap.values_tmc, dshap.vals_loo], num_plot_markers=20,
                                      sources=None, order='d')
        plt.savefig(join('output','cycling',model_family,metric,'plots','descend.pdf'), bbox_inches='tight')
        
        fig = dshap.performance_plots([dshap.values_tmc, dshap.vals_loo], num_plot_markers=20,
                                      sources=None, order='a')
        plt.savefig(join('output','cycling',model_family,metric,'plots','ascend.pdf'), bbox_inches='tight')
        
        values = {}
        values['loo'] = dshap.vals_loo
        values['shapley'] = dshap.values_tmc
        
        pck.dump(values, open(join('output','cycling',model_family,metric,'values.pkl'), 'wb'))
    

# generate shapely and loo statistics for each model in tables
loo_table = []
shp_table = []

for model_family in model_families:
    for metric in metrics:
        if model_family == 'LinearSVC' and metric == 'auc':
            continue
        values = pck.load(open(join('output','cycling',model_family,metric,'values.pkl'), 'rb'))
        loo_stats = stats.describe(values['loo'])
        shp_stats = stats.describe(values['shapley'])
        loo_table.append(['cycling', model_family, metric, loo_stats[1][0], loo_stats[1][1], loo_stats[2], loo_stats[3]])
        shp_table.append(['cycling', model_family, metric, shp_stats[1][0], shp_stats[1][1], shp_stats[2], shp_stats[3]])

loo_table = pd.DataFrame(loo_table, columns=['Data', 'Algorithm', 'Metric', 'Min', 'Max', 'Mean', 'Variance'])
shp_table = pd.DataFrame(shp_table, columns=['Data', 'Algorithm', 'Metric', 'Min', 'Max', 'Mean', 'Variance'])

# print out shapely tables in latex
print('LOOV Statistics')
print(loo_table.to_latex(index=False, float_format="%.2e"))

print('Shapely Statistics')
print(shp_table.to_latex(index=False, float_format="%.2e"))


# generate histograms for shapely values by classifier and metric
for model_family in model_families:
    for metric in metrics:
        if model_family == 'LinearSVC' and metric == 'auc':
            continue
        values = pck.load(open(join('output','cycling',model_family,metric,'values.pkl'), 'rb'))['shapley']
        plt.figure()
        plt.style.use('ggplot')
        plt.hist(values, bins=10)
        plt.savefig(join('output','cycling',model_family,metric,'plots','hist.pdf'))
        
# generate table of shapely values and athletes
values = pck.load(open(join('output','cycling','NB','accuracy','values.pkl'), 'rb'))['shapley'].reshape(-1,1)
athletes = pd.read_csv(join(data_dir, 'train', 'train_activities.csv'))['athlete_id'].to_numpy()[value_data.index].reshape(-1,1)
df = pd.DataFrame(np.concatenate((athletes, values), axis=1), columns=['Athlete', 'Values']).astype({'Athlete' : 'int64'})

# group table by athlete
grouped = df.groupby('Athlete')

# print statistics of grouped values by athlete
print('Athlete Shapely Statistics')
print(grouped.describe()['Values'][['count', 'min', 'max', 'mean', 'std']].to_latex(float_format="%.2e"))

# plot histograms of shapely values grouped by athlete
for group in grouped:
    plt.figure()
    plt.hist(group[1]['Values'])
    plt.savefig(join('output', 'cycling', 'athlete_' + str(group[0]) + '.pdf'), bbox_inches='tight')
    plt.close()

# all value data
X_all = value_data.to_numpy()
y_all = value_labels

# all test data
X_test = eval_data.to_numpy()[:-1162]
y_test = eval_labels[:-1162]


# create a model using most accurate classifier
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

scores = []
all_score = 100*model.fit(X_all, y_all).score(X_test, y_test) # train on all data and test on all data

# take out one athlete at a time from the training data and retrain on the remaining athletes
for group in grouped:
    athlete = group[0]
    X_out = value_data.to_numpy()[athletes.reshape(-1) != athlete, :]
    y_out = value_labels[athletes.reshape(-1) != athlete]
    
    out_score = 100*model.fit(X_out, y_out).score(X_test, y_test)
    scores.append([out_score, out_score - all_score])

scores = pd.DataFrame(scores, columns=['Accuracy (%)', 'Change (%)'], index=[group[0] for group in grouped])
scores.index.name = 'Athlete'

print('Athlete Withholding Accuracies')
print(scores.to_latex(float_format="%.2f"))

