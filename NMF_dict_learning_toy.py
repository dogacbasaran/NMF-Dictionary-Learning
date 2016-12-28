# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 21:01:38 2016
Sparse Non-negative Matrix Factorization
Three divergences d(v_fn || v_hat_fn)
    0 - Itakura-Saito
    1 - Kullbeck-Leibler
    2 - Euclidean

min_(W,H) d(v_fn || v_hat_fn) + lambda sum_n ||h_n||_1

Dictionary Learning example with DIGITS dataset from sklearn
Training Procedure:
    - Learn basis W_i for each number i=0,1,2,...,9
    - Concatenate each W_i to form W_dict = [W_0 W_1 ... W_9]
    - Project each training data onto W_dict to obtain features
    - Train a multinomial logistic regression classifier with these training data
Test Procedure:
    - Project each test data onto W_dict to obtain features
    - Use Logistic Regression Classifier to estimate the digit
    - Compute the score for the test set

    
    
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
X_digits += 1e-12

n_train_samples = np.int(np.floor(0.9 * X_digits.shape[1]))
# Training set
X_train = X_digits[:,:n_train_samples]
y_train = y_digits[:n_train_samples]

# Test set
X_test = X_digits[:,n_train_samples:]
y_test = y_digits[n_train_samples:]


# divergence = 0 -> Itakura-Saito Divergence
# divergence = 1 -> Kullbeck-Leibler Divergence
# divergence = 2 -> Eucledian Divergence
divergence = 2

# Number of basis vectors
K = 20

# The coefficients for L1 and L2 norm constraints on W and H
lambda_1 = 0.1
lambda_2 = 0.
lambda_3 = 0.05
lambda_4 = 0.05

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
    tolerance = 1e-4
    
    one_matrix = np.ones((n_features, n_samples))
    number_of_iterations = 3000
    for iter in range(number_of_iterations):
    
        if divergence == 0: # Itakura-Saito
            #####
            ## IS divergence multiplicative updates
            #####
            V_hat = W_old.dot(H_old)
            # Update W matrix
            W_new = W_old * ((V/V_hat**2).dot(H_old.T)) / ((1/V).dot(H_old.T) + lambda_1 + lambda_3 * W_old)
            
            V_hat = W_new.dot(H_old)        
            # Update H matrix
            H_new = H_old * ((W_new.T).dot((V/V_hat**2))) / ((W_new.T).dot((1/V)) + lambda_2 + lambda_4 * H_old)
        elif divergence==1: # Kullbeck-Liebler
            #####
            ## KL divergence multiplicative updates
            #####
            
            V_hat = W_old.dot(H_old)
            # Update W matrix
            W_new = W_old * ((V/V_hat).dot(H_old.T)) / (one_matrix.dot(H_old.T) + lambda_1 + lambda_3 * W_old)
            
            V_hat = W_new.dot(H_old)        
            # Update H matrix
            H_new = H_old * ((W_new.T).dot((V/V_hat))) / ((W_new.T).dot(one_matrix) + lambda_2 + lambda_4 * H_old)
        elif divergence == 2: # Euclidean
            #####
            ## EUC divergence multiplicative updates
            #####
            
            V_hat = W_old.dot(H_old)
            # Update W matrix
            W_new = W_old * (V.dot(H_old.T)) / (V_hat.dot(H_old.T) + lambda_1 + lambda_3 * W_old)
            
            V_hat = W_new.dot(H_old)        
            # Update H matrix
            H_new = H_old * ((W_new.T).dot((V))) / ((W_new.T).dot(V_hat) + lambda_2 + lambda_4 * H_old)
            
        if np.max(np.max(np.abs(W_new-W_old),axis=0)) < tolerance:
            print('tolerance reached at iteration %d' % iter)
            break
        else:
            if iter == number_of_iterations-1:
                print('tolerance not reached, maximum difference is %f' % np.max(np.max(np.abs(W_new-W_old),axis=0)))
      
            W_old = W_new
            H_old = H_new
            
    if divergence==0:        
        if number==0:        
            W_dict_IS = W_new
        else:
            W_dict_IS = np.concatenate((W_dict_IS,W_new),axis=1)
    elif divergence==1:
        if number==0:        
            W_dict_KL = W_new
        else:
            W_dict_KL = np.concatenate((W_dict_KL,W_new),axis=1)
    elif divergence==2:
        if number==0:        
            W_dict_Euc = W_new
        else:
            W_dict_Euc = np.concatenate((W_dict_Euc,W_new),axis=1)
    
    print(' ')        
# plot_gallery('NMF Bases KL divergence', W.T)

if divergence==0:        
    W_dict = W_dict_IS
elif divergence==1:
    W_dict = W_dict_KL
elif divergence==2:
    W_dict = W_dict_Euc

H_train = (X_train.T).dot(W_dict)

#### The Training Process of the Classifier
logistic = linear_model.LogisticRegression()
logistic.fit(H_train,y_train)

# Extraction of test features
H_test = (X_test.T).dot(W_dict)

print('LogisticRegression score: %f' % logistic.score(H_test, y_test))

y_estimates = logistic.predict(H_test)
errors = np.zeros(10)
for i in range(len(y_test)):
    if y_estimates[i] != y_test[i]:
        errors[y_test[i]] += 1

width = 40/50.0
plt.bar((np.arange(10))-width/2,errors,width=width)
plt.xticks(range(10))
plt.ylim([0, np.max(errors)+1])
plt.xlabel('Digits')
plt.ylabel('Number of errors')
plt.title('The number of errors for each digit')