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
import matplotlib.cm as cm


X1 = np.random.multivariate_normal([5, 5], np.diag([0.3, 0.3]), size=40)
X2 = np.random.multivariate_normal([10, 10], np.diag([0.3, 0.3]), size=40)
X3 = np.random.multivariate_normal([35, 35], np.diag([0.3, 0.3]), size=40)
X4 = np.random.multivariate_normal([40, 40], np.diag([0.3, 0.3]), size=40)
X5 = np.random.multivariate_normal([80, 40], np.diag([0.3, 0.3]), size=40)
X6 = np.random.multivariate_normal([90, 40], np.diag([0.3, 0.3]), size=40)
X = np.vstack([X1, X2,X3, X4,X5, X6])
N, D = X.shape # X is 50* 2 dataset:every data point is a two-dimensional vector
T = 200

# concentration parameter
alpha = 10

# truncation parameter
T = 200

# for the truncated data assume the initial mean and covariance
mean_mu = KMeans(n_clusters = T).fit(X).cluster_centers_[::-1]
cov_mu = np.empty((T, D, D))
for i in np.arange(T):
    cov_mu[i] = np.eye(D)

# parameter for covariance
a_tao = np.ones(T)
b_tao = np.ones(T)

# parameter for v
gamma = alpha * np.ones((T, 2))

# probability that the truncated data would be classified
phi = np.ones((T, N)) / T
Nt = np.sum(phi, axis = 1)

# hyperparamter for ???
a0 = 2
b0 = 3

# iteration times and tolerance
n_iter = 1000
epsilon = 0.00000001


def log_normalize(v, axis=0):
    """Normalized probabilities from unnormalized log-probabilites"""
    v = np.rollaxis(v, axis)
    v = v.copy()
    v -= v.max(axis=0)
    out = logsumexp(v)
    v = np.exp(v - out)
    v += np.finfo(np.float32).eps
    v /= np.sum(v, axis=0)
    return np.swapaxes(v, 0, axis)

#log p(x|mu,tau)
def bound(T,X):
    N, D = X.shape
    bound_x = np.empty((T, N))
    for t in np.arange(T):
        bound_x[t] = np.sum((X - mean_mu[t])**2, axis = 1) + np.trace(cov_mu[t])
    return bound_x


#log p(tau) * p(
def log_lik_x(bound_X):
    likx = np.zeros(bound_X.shape)
    for t in np.arange(T):
        likx[t, :] = .5 * D *(digamma(a_tao[t]) - np.log(b_tao[t]) - np.log(2 * np.pi))
        tao_t = a_tao[t] / b_tao[t]
        likx[t, :] -= .5 * tao_t * bound_X[t]
    return likx

def ELBO(T,alpha,gamma):
    """
        T: truncation level- a variational parameter
        alpha: hyper-paramter for beta distriution
        gamma: hyper-parameter for variational inference
        
        """
    
    # -----------------------------------
    # specify some parameters will be used in the following context
    # -----------------------------------
    lb = 0
    sd = digamma(gamma[:, 0] + gamma[:, 1])
    dg0 = digamma(gamma[:, 0]) - sd
    dg1 = digamma(gamma[:, 1]) - sd
    
    # -----------------------------------
    # pv ~ beta(1,alpha)
    # qv ~ beta(gamma1,gamma2)
    # -----------------------------------
    # E_q[log p(V | 1,alpha)]
    lpv = (gammaln(1 + alpha) - gammaln(alpha)) * T + (alpha - 1) * np.sum(dg1)
    # E_q[log q(V | gamma1, gamma2)]
    lqv = np.sum(gammaln(gamma[:, 0] + gamma[:, 1])  - gammaln(gamma[:, 0]) - gammaln(gamma[:, 1])
                 + (gamma[:, 0] - 1) * dg0 + (gamma[:, 1] - 1) * dg1)
        
                 lb += lpv - lqv
                 
                 # -----------------------------------
                 # p(mu) ~ N(0,I)
                 # q(mu) ~ N(mu,sigma)
                 # -----------------------------------
                 lpmu = 0
                 lqmu = 0
                 
                 for t in np.arange(T):
                     # Eq[log p(mu)]
                     # lpmu += -.5 * (mean_mu[t].dot(mean_mu[t]) + np.trace(cov_mu[t]))
                     
                     lpmu += -.5 * np.dot(np.dot(mean_mu[t],np.linalg.inv(cov_mu[t])),mean_mu[t]) + np.trace(cov_mu[t])
                 
                     sign, logdet = np.linalg.slogdet(cov_mu[t])
                     # Eq[log q(mu | mean_mu, cov_mu)]
                     lqmu += -.5 * sign * logdet
                         lb += lpmu - lqmu

# -----------------------------------
# p(tau) ~ Gamma(1,1)
# q(tau_t) ~ Gamma(a_tao, b_tao)
# -----------------------------------

# Eq[log p(tau)]
lptao = - np.sum(a_tao / b_tao)
# Eq[log q(tau | a_tao, b_tao)]
lqtao = np.sum(-gammaln(a_tao) + (a_tao-1)*digamma(a_tao)
               + np.log(b_tao) - a_tao)
    
    lb += lptao - lqtao
    
    # -----------------------------------
    # p(c) ~ SBP(v)
    # q(c) ~ Discrete(zn | phi_n)
    # -----------------------------------
    phi_cum = np.cumsum(phi[:0:-1, :], axis = 0)[::-1, :]
    lpc = 0
    
    # Eq[log p(Z | V)]
    for t in np.arange(T):
        if t < T - 1:
            lpc += np.sum(phi_cum[t] * dg1[t])
        lpc += np.sum(phi[t] * dg0[t])
    n_phi = phi
    
    # Eq[log q(Z | phi)]
    lqc = np.sum(n_phi * np.log(n_phi))
    
    lb += lpc - lqc
    
    # -----------------------------------
    # X given assignment
    # -----------------------------------
    
    # x
    lpx = 0
    
    # Eq[log p(X)]
    likx = log_lik_x(bound_X)
    lpx = np.sum(phi * likx)
    lb += lpx
    
    return lb

def update_mu(X,T):
    for t in np.arange(T):
        tao_t = a_tao[t] / b_tao[t]
        Nt = np.sum(phi[t])
        cov_mu[t] = np.linalg.inv((tao_t*Nt + 1)*np.eye(D))
        mean_mu[t] = tao_t * cov_mu[t].dot(X.T.dot(phi[t]))

def update_tao(bound_X,T):
    for t in np.arange(T):
        # a_tao[t] = a0 + .5 * D * np.sum(phi[t])
        a_tao[t] = a0 + np.sum(phi[t])
        # b_tao[t] = b0 + .5 * np.sum(np.multiply(phi[t], bound_X[t]))
        b_tao[t] = b0 + np.sum(np.multiply(phi[t], bound_X[t]))

def update_v(X):
    sum_phi = Nt
    gamma[:, 0] = 1 + sum_phi
    phi_cum = np.cumsum(phi[:0:-1, :], axis = 0)[::-1, :]
    gamma[:-1, 1] = alpha + np.sum(phi_cum, axis = 1)

def log_lik_pi(gamma,T):
    sd = digamma(gamma[:, 0] + gamma[:, 1])
    logv = digamma(gamma[:, 0]) - sd
    sum_lognv = np.zeros(T)
    for t in np.arange(1, T):
        sum_lognv[t] = sum_lognv[t-1] + digamma(gamma[t-1, 1]) - sd[t-1]

    likc = logv + sum_lognv
    likc[-1] = np.log(1 - (sum(np.exp(likc[:-1]))))

return likc

def update_c(phi, gamma, T,bound_X):
    likc = log_lik_pi(gamma, T)
    
    likx = log_lik_x(bound_X)
    
    s = likc[:, np.newaxis] + likx
    
    phi = log_normalize(s, axis=0)
    
    return phi, np.sum(phi, axis = 1)

flag = 0
for i in np.arange(n_iter):
    #E STEP
    phi,Nt = update_c(phi, gamma, T,bound_X)
    #M STEP
    update_v(X)
    update_mu(X,T)
    bound_X = bound(T,X)
    update_tao(bound_X,T)
    
    lbs.append(ELBO(T,alpha,gamma))
    
    # if len(lbs) > 1 and 100 * (lbs[-1] - lbs[-2]) / np.abs(lbs[-2]) < epsilon:
    #     break
    flag = flag + 1
    if i % 100 == 0:
        print(i,"iterations")
    if i == n_iter-1:
        print("- finished! -")
clusters = np.argmax(phi, axis=0)

colors = cm.rainbow(np.linspace(0, 1, 6))
plt.figure(figsize=(6,6))
plt.scatter(X[:,0],X[:,1],s = 50, c = colors[clusters])









