# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 21:01:38 2016
Non-negative Matrix Factorization

Dictionary Learning example with digits dataset from sklearn

@author: dbasaran
"""
def plot_gallery(title, images, n_col=2, n_row=3):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(8,8), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

    
    
import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, linear_model

# Import digits dataset
digits = datasets.load_digits();
X_digits = digits.data # Data
y_digits = digits.target # Class

# Plot some samples
#plot_gallery('Some samples', X_digits[0:6,:])

# Data is transposed so that it becomes n_features x n_samples
X_digits = X_digits.T

# Data is scaled to be in the range [0,1]
#X_digits = X_digits/np.max(X_digits,axis=0)
X_digits += 0.000001

n_train_samples = np.int(np.floor(0.9 * X_digits.shape[1]))
# Training set
X_train = X_digits[:,:n_train_samples]
y_train = y_digits[:n_train_samples]

# Test set
X_test = X_digits[:,n_train_samples:]
y_test = y_digits[n_train_samples:]

# Number of basis vectors
K = 20

for number in range(10):
    print('Training for number %d' % number)
    index = np.squeeze(np.asarray(np.where(y_train==number)))
    # Number of features
    n_features = X_train[:,index].shape[0]
    # Number of samples
    n_samples = X_train[:,index].shape[1]

    #### The Training Process of NMF
    W = np.random.uniform(size=(n_features,K))
    H = np.random.uniform(size=(K,n_samples))

    V = X_train[:,index]

    W_old = W;
    H_old = H;
    tolerance = 1e-6
    for iter in range(2000):
    
        #####
        ## KL divergence multiplicative updates
        #####
        # Update W matrix
        W_new = W_old * ((V/W_old.dot(H)).dot(H.T)) / (np.ones((n_features, n_samples)).dot(H.T))
        
        # Update H matrix
        H_new = H_old * ((W_new.T).dot((V/W_new.dot(H_old)))) / ((W_new.T).dot(np.ones((n_features, n_samples))))
    
        if np.max(np.max(np.abs(W_new-W_old),axis=0)) < tolerance:
            print('tolerance reached')
            break
        else:
            W_old = W_new
            H_old = H_new
       
    if number==0:        
        W_dict = W_new
    else:
        W_dict = np.concatenate((W_dict,W_new),axis=1)
        
    print('The size of dictionary is %d\n' % W_dict.shape[1])
# plot_gallery('NMF Bases KL divergence', W.T)

H_train = (X_train.T).dot(W_dict)

#### The Training Process of the Classifier
logistic = linear_model.LogisticRegression()
logistic.fit(H_train,y_train)

# Extraction of test features
H_test = (X_test.T).dot(W_dict)

print('LogisticRegression score: %f' % logistic.score(H_test, y_test))
