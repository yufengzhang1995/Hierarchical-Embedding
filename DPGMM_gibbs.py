import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.misc import logsumexp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
import collections
from scipy import sparse
import numpy as np
from scipy.special import loggamma 
from numpy import random
import scipy.stats as st 
from math import *
from statistics import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# generate synthetic data（draw from multivariate normal distribution）
X1 = np.random.multivariate_normal([5, 5], np.diag([0.5, 0.5]), size=20)
X2 = np.random.multivariate_normal([10, 10], np.diag([0.5, 0.5]), size=20)
X3 = np.random.multivariate_normal([40, 40], np.diag([0.3, 0.3]), size=10)
X4 = np.random.multivariate_normal([50, 50], np.diag([0.6, 0.6]), size=10)
X = np.vstack([X1, X2, X3, X4])
N, D = X.shape # X is 50* 2 dataset:every data point is a two-dimensional vector

# visualize the data
plt.scatter(X[:,0], X[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

#### Define a multivariate t distribution probability density function
def multi_student_pdf(x,d,df,mu,sigma):      # pdf of multivariate student'T distribution 
    term_1 = gamma(1. * (d + df) / 2)
    term_2 = gamma(1. * df / 2) * pow(df, d / 2) * pow(pi, d / 2) * pow(np.linalg.det(sigma),1/2)
    term_3 = pow((1 + (1. / df) * np.dot(np.dot((x - mu),np.linalg.inv(sigma)),(x - mu))),  (1. * (d + df)/2))
    return term_1 / (term_2 * term_3)

def log_student_pdf(x,d,df,mu,sigma):      # log pdf of multivariate student'T distribution 
        term_1 = loggamma(1. * (d + df) / 2)
        term_2 = loggamma(1. * df / 2) + (1. * d / 2) * np.log(1. * df) + (1. * d/2) * np.log(np.pi) + (1. / 2) * np.linalg.det(sigma)
        term_3 = (1. * (d + df) / 2) * np.log(1 + (1. / df) * np.dot(np.dot((x - mu),np.linalg.inv(sigma)),(x - mu)))
        return term_1 - term_2 - term_3

#### Posterior Predictive
def posterior_predictive(mu_0,S_0,data,x_new): # t  degree of freedom nu = nu_n-d+1
    d = D
    N = len(data)
    nu_N = nu_0 + N 
    data_bar = np.mean(data,axis = 0)                            
    lambda_N = lambda_0 + N
    mu_N = (lambda_0 * mu_0 + N * data_bar) / lambda_N                        
    C = (data - mu_0).transpose().dot(data - mu_0)                
    S_N = S_0 + C + lambda_0 / lambda_N * (data_bar - mu_0).transpose().dot(data_bar - mu_0)
    Sigma = S_N * (lambda_N + 1)/(lambda_N * (nu_N - d + 1))
    df = nu_N - d + 1
    return multi_student_pdf(x_new, d, df, mu_N, Sigma)

def log_posterior_predictive(mu_0,S_0,data,x_new): # t  degree of freedom nu = nu_n-d+1
    d = D
    N = len(data)       
    nu_N = nu_0 + N 
    data_bar = np.mean(data,axis = 0)                            
    lambda_N = lambda_0 + N
    mu_N = (lambda_0 * mu_0 + N * data_bar) / lambda_N                        
    C = (data - mu_0).transpose().dot(data - mu_0)                
    S_N = S_0 + C + lambda_0 / lambda_N * (data_bar - mu_0).transpose().dot(data_bar - mu_0)
    Sigma = S_N * (lambda_N + 1)/(lambda_N * (nu_N - d + 1))
    df = nu_N - d + 1
    return log_student_pdf(x_new, d, df, mu_N, Sigma)

def posterior(data,mu_0,S_0): 
    N = len(data)        
    data_bar = np.mean(data,axis = 0) 
    # update cov  
    lambda_N = lambda_0 + N
    nu_N = nu_0 + N  
    C = (data - data_bar).transpose().dot(data - data_bar)                
    S_N = S_0 + C + N * lambda_0 / lambda_N * (data_bar - mu_0).transpose().dot(data_bar - mu_0)
    cov_new = st.invwishart.rvs(nu_N,S_N)               
    # update mean               
    lambda_N = lambda_0 + N
    mu_N = (lambda_0 * mu_0 + N * data_bar) / lambda_N        
    mean_new = st.multivariate_normal.rvs(mu_N,cov_new / lambda_N)              
    result = [mean_new,cov_new]
    return result  

# parameters to estimate
mu = []             # List of 2x1 vector (mean vector of each gaussian) of note: we have already know it is a two-dimensional vector
cov = []

# prior for mu and sigma
mu_0 = np.mean(Y,axis=0)
S_0 = np.eye(D) # priori in inv-wishart distribution

mu.append(np.mean(Y,axis=0)) # initial mean for the cluster
cov.append(S_0) # initial covariance for the cluster


# other 2 parameters in NIW
lambda_0 = 0.01
nu_0 = 4

# prior in the previous CRP
alpha = 100

# Initialize ci for every data vector yi to a random table.
data_assignment = np.zeros([N], dtype=int)       
assignment_matrix = []       
count_in_cluster = []       

# Initialize with ONE cluster
K = 1
assignment_matrix.append(np.ones(N, dtype=int))
data_assignment[:] = 0
count_in_cluster.append(N)

# iterations specification
T = 10

# Collapsed Gibbs Sampler
for it in range(T): # iterations 
    # --------------------------------------------------------
    # Sample from full conditional of assignment from CRP prior (1 step)
    # --------------------------------------------------------
        
    # For each data point, draw the cluster assignment
    for i in range(N): # N data points
        
        # -----------------------------------
        # remove assignment from cluster
        # -----------------------------------
        ith_assignment = data_assignment[i] # get which cluster ith data point belongs to # ith_assignment: the id of the cluster
        assignment_matrix[ith_assignment][i] = 0  # remove the data point from the cluster
        count_in_cluster[ith_assignment] -= 1 # therefore the count in correspongding cluster minus 1
        
        if count_in_cluster[ith_assignment] == 0: # if the number of the data points in the cluster is zero 
            data_assignment[data_assignment > ith_assignment] -= 1 # Update indices by -1
            
            del assignment_matrix[ith_assignment] # delete this cluster
            del count_in_cluster[ith_assignment] # delete this cluster
            del mu[ith_assignment] # delete the corresponding mu
            del cov[ith_assignment] # delete the corresponding cov
            
            K -= 1 # the number of the cluster is down to K - 1

        # -----------------------------------
        # Probability Calculation
        # Draw new assignment zi weighted by CRP prior
        # -----------------------------------
        
        # allocate space for the probability for K + 1 clusters
        probs = np.zeros(K + 1)
        # remove the ith data point 
        data_assignment_minus_i = data_assignment[np.arange(len(data_assignment)) != i] # remove the ith data point

        # -----------------------------------
        # Probs of joining existing cluster
        # -----------------------------------
        for k in range(K):
            # the number all the data points in the kth assignment
            data_assignment_minus_i_k = data_assignment_minus_i[data_assignment_minus_i == k].shape[0] 
            # the data point in the cluster k (excluding i)
            data_K_minus_i = np.delete(X,i,0)[data_assignment_minus_i == k]
            # the data point in the cluster k (including i)
            data_k = X[data_assignment == k]
            # extract the mu and S from their list
            mu_k = mu[k] 
            S_k = cov[k]                 
            # prior 
            crp = data_assignment_minus_i_k / (N + alpha - 1)
            # likelihood to be in an exisitng cluster   (mu_0,S_0,data,x_new)
            likelihood_exist = posterior_predictive(mu_k,S_k,data_K_minus_i,X[i])
            # probability without normalization
            probs[k] = crp * likelihood_exist
       
        # -----------------------------------
        # Probability of creating new cluster
        # -----------------------------------
        
        # prior
        crp = alpha / (N + alpha - 1)    
        # likelihood to be in a new cluster
        likelihood_new =  multi_student_pdf(X[i],D, nu_0 - D + 1,mu_0,S_0) 
        # probability without normalization
        probs[K] = crp * likelihood_new  
        
        # Normalize
        probs /= np.sum(probs)
        # Draw new assignment for i(when the probability goes to max we find the corresponding cluster)
        z = np.random.multinomial(n = 1, pvals = probs).argmax()
        
        # -----------------------------------
        # Update assignment
        # -----------------------------------           
        
        # Update assignment trackers(if we had a new cluster)
        if z == K:
            assignment_matrix.append(np.zeros(N, dtype=int))
            count_in_cluster.append(0)
            mu.append(mu_0)
            cov.append(S_0)
            K += 1
        
        # Update assignment trackers(if we had new data point in existing cluster)
        data_assignment[i] = z
        assignment_matrix[z][i] = 1
        count_in_cluster[z] += 1
        
        
        # -------------------------------------------------
        # Sample from full conditional of cluster parameter
        # -------------------------------------------------

        # -----------------------------------
        # Update mu and S in cluster
        # -----------------------------------
        
        # posterior is NIW
        for k in range(K):
            data_k = X[data_assignment == k]
            
            # update mu
            mu[k] = posterior(data_k,mu_0,S_0)[0]
            # update sigma
            cov[k] = posterior(data_k,mu_0,S_0)[1]
    if  T / (it+1) == 4:
        print("- 25% finished! -")
    if  T / (it+1) == 2:
        print("- 50% finished! -")
    if it == T-1:
        print("- finished! -")

# Collapsed Gibbs Sampler with Log-likelihood
for it in range(T): # iterations 
    # --------------------------------------------------------
    # Sample from full conditional of assignment from CRP prior (1 step)
    # --------------------------------------------------------
        
    # For each data point, draw the cluster assignment
    for i in range(N): # N data points
        
        # -----------------------------------
        # remove assignment from cluster
        # -----------------------------------
        ith_assignment = data_assignment[i] # get which cluster ith data point belongs to # ith_assignment: the id of the cluster
        assignment_matrix[ith_assignment][i] = 0  # remove the data point from the cluster
        count_in_cluster[ith_assignment] -= 1 # therefore the count in correspongding cluster minus 1
        
        if count_in_cluster[ith_assignment] == 0: # if the number of the data points in the cluster is zero 
            data_assignment[data_assignment > ith_assignment] -= 1 # Update indices by -1
            
            del assignment_matrix[ith_assignment] # delete this cluster
            del count_in_cluster[ith_assignment] # delete this cluster
            del mu[ith_assignment] # delete the corresponding mu
            del cov[ith_assignment] # delete the corresponding cov
            
            K -= 1 # the number of the cluster is down to K - 1

        # -----------------------------------
        # Probability Calculation
        # Draw new assignment zi weighted by CRP prior
        # -----------------------------------
        
        # allocate space for the probability for K + 1 clusters
        probs = np.zeros(K + 1)
        # remove the ith data point 
        data_assignment_minus_i = data_assignment[np.arange(len(data_assignment)) != i] # remove the ith data point

        # -----------------------------------
        # Probs of joining existing cluster
        # -----------------------------------
        for k in range(K):
            # the number all the data points in the kth assignment
            data_assignment_minus_i_k = data_assignment_minus_i[data_assignment_minus_i == k].shape[0] 
            # the data point in the cluster k (excluding i)
            data_K_minus_i = np.delete(Y,i,0)[data_assignment_minus_i == k]
            # the data point in the cluster k (including i)
            data_k = Y[data_assignment == k]
            # extract the mu and S from their list
            mu_k = mu[k] 
            S_k = cov[k]                 
            # prior 
            log_crp = np.log(data_assignment_minus_i_k / (N + alpha - 1))
            # likelihood to be in an exisitng cluster   (mu_0,S_0,data,x_new)
            log_likelihood_exist = log_posterior_predictive(mu_k,S_k,data_K_minus_i,Y[i])
            # probability without normalization
            probs[k] = log_crp +  log_likelihood_exist
       
        # -----------------------------------
        # Probability of creating new cluster
        # -----------------------------------
        
        # prior
        log_crp = np.log(alpha / (N + alpha - 1))    
        # likelihood to be in a new cluster
        log_likelihood_new =  log_student_pdf(Y[i],D, nu_0 - D + 1,mu_0,S_0) 
        # probability without normalization
        probs[K] = log_crp + log_likelihood_new  
        
        
        # log to exp
        probs = np.exp(np.array(probs))
        # Normalize
        probs /= np.sum(probs)
        # Draw new assignment for i(when the probability goes to max we find the corresponding cluster)
        z = np.random.multinomial(n = 1, pvals = probs).argmax()
        
        # -----------------------------------
        # Update assignment
        # -----------------------------------           
        
        # Update assignment trackers(if we had a new cluster)
        if z == K:
            assignment_matrix.append(np.zeros(N, dtype=int))
            count_in_cluster.append(0)
            mu.append(mu_0)
            cov.append(S_0)
            K += 1
        
        # Update assignment trackers(if we had new data point in existing cluster)
        data_assignment[i] = z
        assignment_matrix[z][i] = 1
        count_in_cluster[z] += 1
        
        
        # -------------------------------------------------
        # Sample from full conditional of cluster parameter
        # -------------------------------------------------

        # -----------------------------------
        # Update mu and S in cluster
        # -----------------------------------
        
        # posterior is NIW
        for k in range(K):
            data_k = Y[data_assignment == k]
            
            # update mu
            mu[k] = posterior(data_k,mu_0,S_0)[0]
            # update sigma
            cov[k] = posterior(data_k,mu_0,S_0)[1]
    if it % 10 == 0:
        print(it,"iterations")
    if it == T-1:
        print("- finished! -")




        

for k in range(K):
    print('{} data in cluster-{}'.format(count_in_cluster[k], k))

colors = cm.rainbow(np.linspace(0, 1, len(count_in_cluster)))
plt.figure(figsize=(5,5))
plt.scatter(X[:,0],X[:,1],s = 50, c = colors[data_assignment])


# PCA reduction and visualization（for embedding data）
def pca_df2(x_subset):
    pca = PCA(n_components = 4)
    pca_result = pca.fit_transform(x_subset)
    pca_df = pd.DataFrame(columns = ['pca1','pca2','pca3','pca4'])
    pca_df['pca1'] = pca_result[:,0]
    pca_df['pca2'] = pca_result[:,1]
    pca_df['pca3'] = pca_result[:,2]
    pca_df['pca4'] = pca_result[:,3]
    top_two_comp = pca_df[['pca1','pca2','pca3']]
    return top_two_comp


















