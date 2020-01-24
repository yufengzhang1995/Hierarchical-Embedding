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

def spectral_k_means(N,n,G):
    """
    N = orginal size
    n = target size (not too small)
    G : parse in file to create graph
    return G coarsened from N node to n node in the updated graph
    """
    # eigendecomposition of Laplacian matrix
    node_degree = [item[1] for item in list(G.degree(G.nodes()))]
    D = np.diag(1 / np.sqrt(node_degree))
    W = nx.to_numpy_matrix(G)
    L = np.identity(N) - D.dot(W).dot(D)
    eigenValues, eigenVectors = np.linalg.eig(L)
    
    # sort eigenvalues and get corresponding eigenvectors
    idx = eigenValues.argsort()[::1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    # get k1 and k2 satifitying the condition
    k1 = 0
    for i in np.arange(N):
        if (i + 1 < n and eigenValues[i] <= 1 and eigenValues[N - n + i + 1] >= 1):
            k1 = i
            break
    k2 = N - n + k1
    
    # set a large cost
    cost = 10**20
    # record the minimum of cost and corresponding k1
    cost_rec = []
    k1_rec = []
    while k1 + 1 < n and eigenValues[k1] <= 1 and eigenValues[k2 + 1] >= 1:
        # construct Uk
        U_k1 = np.zeros((N,n))
        for i in np.arange(k1):
            U_k1[:,i] = (np.squeeze(np.asarray(eigenVectors[0:k1][i])))
        for i in np.arange(N - k2):
             U_k1[:,k1 + i] = np.squeeze(np.asarray(eigenVectors[k2:N][i]))
        
        # kmeans clustering
        km_cluster = KMeans(n_clusters = n, random_state=0).fit(U_k1) # not sure whether it is correct 
        # get cost for the k1 and k2
        cost_temp = km_cluster.inertia_
        if cost_temp < cost:
            k1_rec.append(k1)
            cost = cost_temp
        cost_rec.append(cost)
        k1 = k1 + 1 
        k2 = N - n + k1
    
    # return the minimum of cost and corresponding k1 and cluster result
    k1_min = k1_rec[-1]
    U_k = np.zeros((N,n))
    for i in np.arange(k1):
        U_k[:,i] = (np.squeeze(np.asarray(eigenVectors[0:k1][i])))
    for i in np.arange(N - k2):
        U_k[:,k1 + i] = np.squeeze(np.asarray(eigenVectors[k2:N][i]))
    k_cluster = KMeans(n_clusters = n, random_state=0).fit(U_k)
    result = k_cluster.fit_predict(U_k)
    
    # add cluster labels to every node
    labels = []
    nx.set_node_attributes(G,labels,'labels')
    nodesList = list(G.nodes)
    for node in nodesList:
        idx = nodesList.index(node)
        G.nodes[node]['labels'] = result.tolist()[idx]
    
    # create hierarchical list to store nodes with same label
    lists = [[] for _ in range(n)]
    for node in nodesList:
        for i in np.arange(n):
            if G.nodes[node]['labels'] == i:
                lists[i].append(node)
    
    # create supernode
    supernodes = lists

    G_updated = G
    for supernode in supernodes:
        for node in supernode[1:]:
            G_updated = nx.contracted_nodes(G_updated, supernode[0], node)
    return G_updated
