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


def get_csv(path):
    if not isdir(path):
        print('Error: {} is not a directory'.format(path))
        return
    files = [join(path,file) for file in listdir(path) if file[-4:] == '.csv']
    if len(files) == 0:
        print('Error: No CSV files in {}'.format(path))
        return
    elif len(files) > 1:
        print('Warning: Multiple CSV files in {}'.format(path))
        print('Choose CSV file')
        for idx,file in enumerate(files):
            print('{}:\t{}'.format(idx, file))
        return files[int(input('Enter Option [0-{}]: '.format(len(files)-1)))]    
        
    
    return files[0]     

data_dir = join('data', 'cycling')

train_data = pd.read_csv(get_csv(join(data_dir,'train')))
test_data = pd.read_csv(get_csv(join(data_dir,'test')))

columns = ['moving_time','avg_speed','max_speed','elevation_gain','avg_hr','max_hr','calories','avg_cadence','max_cadence']
names = ['Moving Time','Avg Speed','Max Speed','Elevation Gain','Avg HR','Max HR','Calories','Avg Cadence','Max Cadence']

target = 'avg_power'
tar_name = 'Avg Power'

train_breaks = jenks_breaks(train_data[target], nb_class=2)
train_breaks[0] = -inf
train_breaks[-1] = inf
train_labels = np.ravel(np.digitize(train_data[target], train_breaks))-1
test_labels = np.ravel(np.digitize(test_data[target], train_breaks))-1

train_data = train_data[columns]
test_data = test_data[columns]

train_data.to_pickle(join(data_dir,'train','train_data.pkl'))
test_data.to_pickle(join(data_dir,'test','test_data.pkl'))

np.save(join(data_dir,'train','train_labels.npy'), train_labels)
np.save(join(data_dir,'test','test_labels.npy'), test_labels)

from sklearn.model_selection import StratifiedShuffleSplit

value_idx, eval_idx = list(StratifiedShuffleSplit(n_splits=1, train_size=0.1).split(train_data, train_labels))[0]

value_data = train_data.loc[value_idx,:]
eval_data = train_data.loc[eval_idx,:]

value_labels = train_labels[value_idx]
eval_labels = train_labels[eval_idx]

value_data.to_pickle(join(data_dir, 'train', 'value_data.pkl'))
eval_data.to_pickle(join(data_dir, 'train', 'eval_data.pkl'))

np.save(join(data_dir, 'train', 'value_labels.npy'), value_labels)
np.save(join(data_dir, 'train', 'eval_labels.npy'), eval_labels)