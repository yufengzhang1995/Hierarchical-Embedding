import networkx as nx
import random
import numpy as np
import pandas as pd
import scipy as sp
from node2vec import Node2Vec
from sklearn import manifold, datasets
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


sizes = [40,40,40,40]
p = [[0.9,0.01,0.01,0.01],
     [0.01,0.9,0.01,0.01],
     [0.01,0.01,0.9,0.01],
     [0.01,0.01,0.01,0.9]]
G = nx.generators.community.stochastic_block_model(sizes,p,seed = 42)
labels = np.array(["1"]* 40 + ["2"]* 40 + ["3"]* 40 + ["4"]* 40)
N = len(G)
n = 4



def multilevel_graph_coarsening(G0,s,n,i):
    while s > n:
        
        nodes = list(G0.nodes)
        
        A = nx.adjacency_matrix(G0)
        D = np.squeeze(np.asarray(A.sum(axis = 1)))
        
        ds = np.zeros((s,s))
        
        for node in nodes:
            m = nodes.index(node)
            neighbors = G0.neighbors(node)
            for neighs in neighbors:
                q = nodes.index(neighs)
                ds[m,q] = np.linalg.norm(np.asarray(A.todense()[m,]) / D[m] -  np.asarray(A.todense()[q,]) / D[q], ord=1)
        
        ds = ds + 100 * np.identity(s)
        value = ds.min()
        min_i, min_j = np.where(ds == value)
        index_node_i = min_i.tolist()
        index_node_j = min_j.tolist()

        new_G = nx.contracted_nodes(G0, nodes[index_node_i[0]], nodes[index_node_j[0]])
        G0 = new_G
        s = s - 1
        i = i + 1

return G0
G_coarse_multilevel = multilevel_graph_coarsening(G,N,n,1)
