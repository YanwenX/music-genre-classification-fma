#!/usr/bin/python

import scipy.io as spio
import numpy as np
import math
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# load data
X_train_and_val = spio.loadmat('X_train_and_val_2.mat')
y_train_and_val = spio.loadmat('y_train_and_val_6class.mat')
X_test = spio.loadmat('X_test_2.mat')
y_test = spio.loadmat('y_test_6class.mat')

# extract design matrix and labels
for key_X in X_train_and_val.keys():
    if isinstance(X_train_and_val[key_X], np.ndarray):
        Xtrain_and_val = X_train_and_val[key_X]

for key_y in y_train_and_val.keys():
    if isinstance(y_train_and_val[key_y], np.ndarray):
        ytrain_and_val = y_train_and_val[key_y]

for key_X in X_test.keys():
    if isinstance(X_test[key_X], np.ndarray):
        Xtest = X_test[key_X]

for key_y in y_test.keys():
    if isinstance(y_test[key_y], np.ndarray):
        ytest = y_test[key_y]

# split training and validation set
size_train = 3200
size_val = 500
labels = np.unique(ytrain_and_val)
N_class = len(labels)

Xtrain = np.zeros((N_class * size_train, Xtrain_and_val.shape[1]))
ytrain = np.zeros((N_class * size_train, 1))
Xval = np.zeros((N_class * size_val, Xtrain_and_val.shape[1]))
yval = np.zeros((N_class * size_val, 1))

for i in range(N_class):
    Xtrain[i*size_train : (i+1)*size_train, :] =\
    Xtrain_and_val[i*(size_train+size_val) : i*(size_train+size_val)+size_train, :]
    ytrain[i*size_train : (i+1)*size_train] =\
    ytrain_and_val[i*(size_train+size_val) : i*(size_train+size_val)+size_train]

    Xval[i*size_val : (i+1)*size_val, :] =\
    Xtrain_and_val[i*(size_train+size_val)+size_train : (i+1)*(size_train+size_val), :]
    yval[i*size_val : (i+1)*size_val] =\
    ytrain_and_val[i*(size_train+size_val)+size_train : (i+1)*(size_train+size_val)]

# data preprocessing (standardization)
Xtrain_scaled = preprocessing.scale(Xtrain)
Xytrain = np.concatenate((Xtrain_scaled, ytrain), axis = 1)
np.random.seed(1)
np.random.shuffle(Xytrain)
Xtrain_scaled = Xytrain[:, :-1]
ytrain = Xytrain[:, -1]

Xval_scaled = preprocessing.scale(Xval)
Xtest_scaled = preprocessing.scale(Xtest)

# set regularizer and multi_class mode
regularizer = np.logspace(-3.0, 3.0, num = 10)

# train the model
print("OVR, L2-regularizer, LBFGS solver, max iteration = 100")
for c in regularizer:
    classifier = LogisticRegression(penalty='l2', solver='lbfgs', C=c, \
                                    max_iter=100, multi_class='ovr', \
                                    random_state=1)
    classifier.fit(Xtrain_scaled, ytrain)

    # print the training score
    print("training score: %.5f, validation score: %.5f (C = %.5f)" %\
          (classifier.score(Xtrain_scaled, ytrain),\
           classifier.score(Xval_scaled, yval), c))

