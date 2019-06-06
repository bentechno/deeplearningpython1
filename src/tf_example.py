#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 17:02:38 2018

@author: mbaye
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
Nclass = 500 
D = 2 # dimension of output
M = 3 # hidden layer size
K = 3 # nmber of classes
X1 = np.random.randn(Nclass,D) + np.array([0, -2])
X2 = np.random.randn(Nclass,D) + np.array([2, 2])
X3 = np.random.randn(Nclass,D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])
Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
N = len(Y)

T = np.zeros ((N, K))
for i in xrange(N):
    T[i,Y[i]] = 1

plt.scatter(X[:,0],X[:,1],c=Y, s=100, alpha=0.5)
plt.show()

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
def forward (X, W1, b1, W2, b2):
    Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)