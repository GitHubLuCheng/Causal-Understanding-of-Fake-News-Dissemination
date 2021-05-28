import pickle
import pandas as pd
import networkx as nx
import numpy as np
from networkx.algorithms import bipartite
import random
from sklearn.decomposition import LatentDirichletAllocation as LDA

"""Creat user-fakenews bipartite graph"""

f = open("Data/all_users_attribute.pkl","rb")
users=pickle.load(f)
f.close()
edges_list=[]
news_list=[]
news=pd.read_csv('Data/dEFEND data(5)/dEFEND data/gossipcop_content_no_ignore.tsv', sep='\t', header=0)
news=news.set_index('id').T.to_dict('list')

for item in users.keys():
    user_news=users[item]['news_ids']
    for new_id in user_news:
        if new_id not in news.keys():
            continue
        elif news[new_id][0]==1:
            news_list.append(new_id)
            edges_list.append((item,new_id))
news_list=list(dict.fromkeys(news_list))
B=nx.Graph()
B.add_nodes_from(users.keys(),bipartite=0)
B.add_nodes_from(news_list,bipartite=1)
B.add_edges_from(edges_list)

print(B.number_of_edges())
B.remove_nodes_from(list(nx.isolates(B)))
print(B.number_of_nodes())

print(nx.is_connected(B))

edges_list=list(B.edges)
user_nodes=[]
news_list=[]
for e in edges_list:
    user_nodes.append(e[0])
    news_list.append(e[1])

user_nodes=list(set(user_nodes))
news_list=list(set(news_list))
print(len(news_list))
print(len(user_nodes))
#degrees = [val for (node, val) in B.degree()]
graphs = list(nx.connected_component_subgraphs(B))
print(len(graphs))

probability={}
news_map={}
users_map={}
id_new=0
id_u=0
f1=open('Data/Gossip/users_list.txt','w')
f2=open('Data/Gossip/news_list.txt','w')
f1.write('org_id remap_id\n')
f2.write('org_id remap_id\n')
f3=open('Data/Gossip/user_label.txt','w')
for (node, val) in B.degree():
    if node in news_list:
        news_map[node]=id_new
        id_new+=1
        f2.write(node+' '+str(id_new)+'\n')
        probability[node]=1./val
    elif node in user_nodes:
        users_map[node] = id_u
        id_u += 1
        f1.write(node + ' ' + str(id_u) + '\n')
        f3.write(node + ' '+str(val)+'\n')

f1.close()
f2.close()
f3.close()

N=len(edges_list)
N_test=int(N*0.1)
p_edges=[]
for e in edges_list:
    p_edges.append(probability[e[1]])

sum_p=sum(p_edges)
prob=[p/sum_p for p in p_edges]
test_id=np.random.choice(range(N),N_test,replace=False,p=prob)
test=[edges_list[i] for i in test_id]
test_users_interactions={}
for e in test:
    edges_list.remove(e)
    B.remove_edge(e[0],e[1])
    if users_map[e[0]] not in test_users_interactions.keys():
        test_users_interactions.setdefault(users_map[e[0]],[str(news_map[e[1]])])
    else:
        test_users_interactions[users_map[e[0]]].append(str(news_map[e[1]]))

f1=open('Data/Gossip/test.txt','w')
f_pscore=open('Data/Gossip/test-pscore.txt','w')
for u in test_users_interactions.keys():
    f1.write(str(u)+' '+' '.join(test_users_interactions[u])+'\n')
    for n in test_users_interactions[u]:
        f_pscore.write(str(u)+' '+n+'\n')
f_pscore.close()
f1.close()

graphs = list(nx.connected_component_subgraphs(B))
print(len(graphs))
train_users_interactions={}
for e in edges_list:
    if users_map[e[0]] not in train_users_interactions.keys():
        train_users_interactions.setdefault(users_map[e[0]], [str(news_map[e[1]])])
    else:
        train_users_interactions[users_map[e[0]]].append(str(news_map[e[1]]))

f2=open('Data/Gossip/train.txt','w')
f_pscore=open('Data/Gossip/train-pscore.txt','w')
for u in train_users_interactions.keys():
    f2.write(str(u)+' '+' '.join(train_users_interactions[u])+'\n')
    for n in train_users_interactions[u]:
        f_pscore.write(str(u) + ' ' + n + '\n')
f_pscore.close()
f2.close()

news_used = {k: news[k][1] for k in news_list}

lda = LDA(n_components=5, random_state=0,n_jobs=-1)
n_features=2000
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

vectorizer = CountVectorizer(max_df=0.5, max_features=n_features,
                                 min_df=5, stop_words='english')
X = vectorizer.fit_transform(news_used.values()).toarray()

X_topic=lda.fit_transform(X)
np.savetxt('Data/Gossip/items_lda_attribute.txt',X_topic)

f3=open('Data/Gossip/users_attribute.txt','w')
key_to_remove=["news_ids", "label",'embedding']
for u in users.keys():
    if u in user_nodes:
        attributes=[]
        keys=users[u]
        for k in keys:
            if k not in key_to_remove:
                values=users[u][k]
                if isinstance(values,int):
                    attributes.append(values)
                else:
                    attributes+=values.tolist()
        attributes = [str(a) for a in attributes]
        f3.write(str(users_map[u])+' '+' '.join(attributes)+'\n')
