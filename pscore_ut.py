"""
Codes for preprocessing real-world datasets and computing propensity scores used in the experiments
"""
import codecs
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import collections
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.model_selection import train_test_split


"""Load and Preprocess datasets."""
# load dataset.
col = {0: 'user', 1: 'item'}
with open(f'Data/Gossip/train-pscore.txt', 'r') as f:
    data_train = pd.read_csv(f, delimiter=' ', header=None)
    data_train.rename(columns=col, inplace=True)
with open(f'Data/Gossip/test-pscore.txt', 'r') as f:
    data_test = pd.read_csv(f, delimiter=' ', header=None)
    data_test.rename(columns=col, inplace=True)
num_users, num_items = data_train.user.max(), data_train.item.max()
#for _data in [data_train, data_test]:
 #   _data.user, _data.item = _data.user - 1, _data.item - 1

# train-test, split
train, test = data_train.values, data_test.values
combine=np.vstack((train,test))
users = np.loadtxt(
            'Data/Gossip/users_attribute.txt',
            delimiter=' ')
followers = users[:, 5]
item_user=dict()
# estimate pscore
for i in range(combine.shape[0]):
    u=combine[i,0]
    it=combine[i,1]
    pu = followers[u]
    if it not in item_user.keys():
        item_user.setdefault(it,pu)
    else:
        item_user[it]+=pu

item_user_sorted=collections.OrderedDict(sorted(item_user.items()))
item_freq=np.asarray(list(item_user_sorted.values()))
pscore = (item_freq / np.max(item_freq)) ** 0.5

path = Path(f'Data/Gossip')
path.mkdir(parents=True, exist_ok=True)
#np.save(str(path / 'train.npy'), arr=train.astype(np.int))
#np.save(str(path / 'val.npy'), arr=val.astype(np.int))
#np.save(str(path / 'test.npy'), arr=test.astype(np.int))
np.save(str(path / 'pscore_ut.npy'), arr=pscore)
