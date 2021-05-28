import pandas as pd
import pickle
import csv
import numpy as np

def mergedict(a,b):
    a.update(b)
    return a

news=pd.read_csv('Data/dEFEND data(5)/dEFEND data/gossipcop_content_no_ignore.tsv', sep='\t', header=0)
explict_real=pd.read_csv("Data/explicit_data/explicit_data/gossipcop_real_all_user_explicit_features.csv")
explict_fake=pd.read_csv("Data/explicit_data/explicit_data/gossipcop_fake_all_user_explicit_features.csv")
with open("Data/m3_data_out/m3_data_out/gossipcop_real_m3_all_user_info.pkl", 'rb') as pickle_file:
    m3_real=pickle.load(pickle_file)
with open("Data/m3_data_out/m3_data_out/gossipcop_fake_m3_all_user_info.pkl", 'rb') as pickle_file:
    m3_fake=pickle.load(pickle_file)

user_attributes={}
explict_real['news_ids'] = explict_real['news_ids'].str.strip().str.split('|')
explict_real.set_index("user_name", drop=True, inplace=True)
ex_real_dict = explict_real.to_dict(orient="index")
explict_fake['news_ids'] = explict_fake['news_ids'].str.strip().str.split('|')
explict_fake.set_index("user_name", drop=True, inplace=True)
ex_fake_dict = explict_fake.to_dict(orient="index")

ex_user_dict=ex_real_dict
for key in ex_fake_dict.keys():
    if key not in ex_user_dict.keys():
        ex_user_dict[key]=ex_fake_dict[key]
    else:
        ex_user_dict[key]['news_ids']+=ex_fake_dict[key]['news_ids']

m3_user_dict=m3_real
for key in m3_fake.keys():
    if key not in m3_user_dict.keys():
        m3_user_dict[key]=m3_fake[key]

for key in m3_user_dict.keys():
    for feature in m3_user_dict[key].keys():
        a=np.array(list(m3_user_dict[key][feature].values()))
        a[np.where(a==np.max(a))] = 1
        a[np.where(a <1)] = 0
        m3_user_dict[key][feature]=a

print(len(ex_user_dict.keys()))
print(len(m3_user_dict.keys()))
user_attributes=ex_user_dict.copy()
for key in ex_user_dict.keys():
    if key not in m3_user_dict.keys():
        del user_attributes[key]
        continue
    user_attributes[key].update(m3_user_dict[key])

import copy
new_user_attributes=copy.deepcopy(user_attributes)
for user in user_attributes.keys():
    news_ids=user_attributes[user]['news_ids']
    user_news= news[news['id'].isin(news_ids)]
    labels=user_news['label'].values
    if labels.shape[0]==0:
        del new_user_attributes[user]
        continue
    label=np.mean(labels)
    new_user_attributes[user]['label']=label

print(len(new_user_attributes.keys()))
f = open("Data/all_users_attribute.pkl","wb")
pickle.dump(new_user_attributes,f)
f.close()
