#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 02:00:49 2022

@author: scott
"""

import numpy as np
import pandas as pd

from sklearn import datasets
import scipy

x1 = [-0.001271673,-0.008055735, 0.997513967, None, -0.466050483,
-2.831182321,-0.295235185, 0.321523554, 0.452224977, None]
x2 = [1.094754192, 0.516446048,-0.038293932,-2.447391202,-0.735928229,
-0.004057757, None,-0.657263241, 0.055907470, None]
x3 = [0,np.nan,1,0,0,2,1,2,0,1]

is_categorical=[False,False,True]
x=[x1,x2,x3]


iris = datasets.load_iris()
x = [col.reshape(col.shape[0]) for col in np.hsplit(iris.data, iris.data.shape[1])]
labels = iris.feature_names
x.reverse() ; labels.reverse()
x.append(iris.target)
labels.append("species")
x.reverse() ; labels.reverse()
is_categorical=[True,False,False,False,False]


def flatten_list(lst):
    n=lst[0].shape[0]
    cols = []
    for x in lst:
        if x.ndim == 1: 
            cols.append(x.reshape(n))
        else:
            for i in range(x.shape[1]):
                cols.append(x[:,i].reshape(n))
    return np.stack(cols, axis=1)


#def transform(x, w, y, n, p, q, Q, dim, ordered):

p=len(x) #3
n=len(x[0]) #10

Xj=[]; wj=[] ; qj=[] ; j=0

def is_none(item):
    return item is None or np.isnan(item) or str(item) == "NA"

for xi, cat in zip(x, is_categorical):
    wi = np.array([0 if is_none(x_) else 1 for x_ in xi])
    xi=[0 if w_ == 0 else x_ for w_,x_ in zip(wi,xi)]
    wj.append(wi)
    if cat: 
        Z = pd.get_dummies(xi).to_numpy()
        #todo: GENERALIZE C
        C=np.array([0,0,1,0,1,1]).reshape(3,2)
        print("C:\n",C)
        X_=Z.dot(C)
        Xj.append(X_)
        for col in range(X_.shape[1]):
            qj.append(j)
    else: 
        Xj.append(np.array(xi).reshape(n,1))
        qj.append(j)
    j+=1


W = np.stack(wj, axis=1)
d = np.sum(W, axis=1)
if sum(d==0) > 0:
    raise ValueError("One or more records is completely missing data.")

Wpad = np.stack([wj[i] for i in qj], axis=1)
X=flatten_list(Xj)

# High-dimensional data visualisation: The textile plot
# appendix B defines A12, A22 for categorical

# A11 ------------------------
d11 = np.ones(n).transpose().dot(W)
A11 = -W.transpose().dot(W/d.reshape(n, 1)) + np.diag(d11)

# A12 ------------------------
A12= Wpad.transpose().dot(Wpad*X/d.reshape(n, 1))
d12 = np.diag(Wpad.transpose().dot(X))
d12 = np.diag(d12)

to_remove = [] ; s=set()
for j,q_ in enumerate(qj):
    if q_ in s:
        to_remove.append(j)
        d12[q_] += d12[j]
    s.add(q_)

A12 = np.delete(A12, to_remove, axis=0) - np.delete(d12, to_remove, axis=0)

# A22 ------------------------
d22_full = X.transpose().dot(Wpad)*Wpad.transpose().dot(X)/np.ones(n).reshape(1,n).dot(Wpad)
d22 = np.zeros((d22_full.shape))
skip = 0       
for k,j in enumerate(qj):
    j=j+skip
    if j != k: skip+=1
    while j <= k:
        d22[j,k] = d22_full[j,k] ; d22[k,j] = d22_full[k,j]
        j+=1

A22 = -(Wpad*X).transpose().dot(Wpad*X/d.reshape(n, 1)) + d22

# B ------------------------
B_full = np.dot(X.transpose(), Wpad*X) - d22_full
B = np.zeros((B_full.shape))

skip = 0       
for k,j in enumerate(qj):
    j=j+skip
    if j != k: skip+=1
    while j <= k:
        B[j,k] = B_full[j,k] ; B[k,j] = B_full[k,j]
        j+=1

#print("a11:\n", A11.round(3))
#print("a12:\n", A12.round(3))
#print("a22:\n", A22.round(3))

A11inv = np.linalg.pinv(A11, rcond=1e-10)

A=(np.dot(A12.transpose(), A11inv).dot(A12) - A22)
print("A:\n", A.round(3))
print("B:\n", B.round(3))

eigenvectors, eigenvalues, _ = scipy.linalg.lapack.dsygv(A, B, uplo="U")
print("eigenvectors:\n", eigenvectors.round(4))
#print("eigenvalues:\n", eigenvalues.round(4))
# in ascending order, last col has biggest eigenvalue
beta = eigenvectors[:,-1]

#z=np.zeros(A11.shape[0])
alpha=np.dot(A11inv, A12).dot(beta) #+ (np.ones(A11.shape) - np.dot(A11inv, A11)).dot(z)

print("beta:\n", beta)
print("alpha:\n", alpha)


yj=[]

for j,xj in enumerate(Xj):
    k=[i for i,q in enumerate(qj) if q == j]
    yj.append(xj.dot(beta[k]) + alpha[j])

Y=np.stack(yj, axis=1)
y0=(W*Y).sum(axis=1)/Y.shape[1]
yj = [y0] + yj
Y=np.stack(yj, axis=1)

#print("Y:\n", Y.round(5))

'''
print("------------------------------------------------------------------------------------")
print("no missing")

n=X.shape[0]
p=5

XTX = X.transpose().dot(X)
ones = np.ones(n).reshape(n, 1)
XT11TX = X.transpose().dot(ones).dot(ones.transpose()).dot(X)
A = (XTX - XT11TX/n)/p

B_full = XTX - XT11TX/n
B = np.zeros((B_full.shape))

skip = 0
for k,j in enumerate(qj):
    j=j+skip
    if j != k: skip+=1
    while j <= k:
        B[j,k] = B_full[j,k] ; B[k,j] = B_full[k,j]
        j+=1

print("A:\n", A.round(3))
print("B:\n", B.round(3))

EM, beta, res = scipy.linalg.lapack.dsygv(A,B)

print(beta.transpose().dot(B).dot(beta))

z=np.zeros(A11.shape[0])
alpha=np.dot(A11inv, A12).dot(beta) + (np.ones(A11.shape) - np.dot(A11inv, A11)).dot(z)
'''
