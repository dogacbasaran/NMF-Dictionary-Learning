# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 21:01:38 2016
Non-negative Matrix Factorization

V = WH

@author: dbasaran
"""
def plot_gallery(title, images, n_col=2, n_row=3):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(64,64), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

    
    
import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import fetch_olivetti_faces

from sklearn import decomposition

rng = np.random.RandomState(0)

dataset = fetch_olivetti_faces(shuffle=True,random_state=rng)

faces = dataset.data

n_samples, n_features = faces.shape

K = 6

W = np.random.uniform(size=(n_features,K))
H = np.random.uniform(size=(K,n_samples))

V = faces.T

for iter in range(200):
    
    #####
    ## KL divergence multiplicative updates
    #####
    # Update W matrix 
    W = W * ((V/W.dot(H)).dot(H.T)) / (np.ones((n_features, n_samples)).dot(H.T))
    
    # Update H matrix
    H = H * ((W.T).dot((V/W.dot(H)))) / ((W.T).dot(np.ones((n_features, n_samples))))

plot_gallery('NMF Bases KL divergence', W.T)

nmf_toy = decomposition.NMF(n_components=K, init='nndsvda', tol=5e-3)

nmf_toy.fit(faces);

components_ = nmf_toy.components_

plot_gallery('NMF Bases Fronobis', components_)

