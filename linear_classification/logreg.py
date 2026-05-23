import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def cal_gradient(X,w,y):
    m=y.size
    return (X.T@(sigmoid(X@w)-y))/m

def grad_des(X,y,alpha,reps):
    X_b=np.c_[np.ones((X.shape[0],1)),X]
    w=np.zeros(X_b.shape[1])

    for i in range(reps):
        grad=cal_gradient(X_b,w,y)
        w= w-(grad*alpha)
    
    return w

def giv_prob(X,w):
    X_b=np.c_[np.ones((X.shape[0],1)),X]
    return sigmoid(X_b@w)

def giv_result(X,w):
    return (giv_prob(X,w)>=0.5).astype(int)

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X,y=load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)

w_hat=grad_des(X_train_scaled,y_train,0.22,1000)

y_pred_train=giv_result(X_train_scaled,w_hat)
y_pred_test=giv_result(X_test_scaled,w_hat)

train_acc=accuracy_score(y_train,y_pred_train)
test_acc=accuracy_score(y_test,y_pred_test)

print(train_acc)
print(test_acc)