import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('train.csv')
m,n=data.shape
#print(data.head)
data=np.array(data)
data=data.T
Y_train=data[0]
X_train=data[1:n]/255.0

test=pd.read_csv('test.csv')
l,k=test.shape
test=np.array(test)
test=test.T
X_test=data[0:k]/255.0

def parameters():
    w1 = np.random.randn(10, 784) * 0.01
    b1=np.random.rand(10,1)
    w2 = np.random.randn(10, 10) * 0.01
    b2=np.random.rand(10,1)
    return w1,b1,w2,b2

def Relu(Z):
    return np.maximum(0,Z)

def softmax(z):
    exp_Z = np.exp(z - np.max(z, axis=0, keepdims=True))  # Prevents overflow
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def for_prop(w1,b1,w2,b2,X):
    z1=w1.dot(X)+b1
    a1=Relu(z1)
    z2=w2.dot(a1)+b2
    a2=softmax(z2)
    return z1,a1,z2,a2

def one_hot(Y):
    one_hot_Y=np.zeros((Y.size,10))
    one_hot_Y[np.arange(Y.size), Y] = 1 #a witty step necessarily review this
    one_hot_Y=one_hot_Y.T
    return one_hot_Y

def back_prop(X,z1,w1,a1,z2,w2,a2,Y):
    m=Y.size
    one_hot_Y=one_hot(Y)
    dz2=a2-one_hot_Y
    dw2=1/m*dz2.dot(a1.T)
    db2=1/m*np.sum(dz2,axis=1, keepdims=True)
    dz1=w2.T.dot(dz2)*(z1>0)
    dw1=1/m*dz1.dot(X.T)
    db1=1/m*np.sum(dz1,axis=1, keepdims=True)
    return dw1,db1,dw2,db2

def upd_params(w1,w2,b1,b2,dw1,dw2,db1,db2,alpha):
    w1=w1-alpha*dw1
    w2=w2-alpha*dw2
    b1=b1-alpha*db1
    b2=b2-alpha*db2
    return w1,b1,w2,b2

def get_pred(a2):
    return np.argmax(a2,0)

def accu(predictions,Y):
    print(predictions,Y)
    return np.sum(predictions==Y)/Y.size

def grad_des(X,Y,epochs,alpha):
    w1,b1,w2,b2=parameters()
    for i in range(epochs):
        z1,a1,z2,a2=for_prop(w1,b1,w2,b2,X)
        dw1,db1,dw2,db2=back_prop(X,z1,w1,a1,z2,w2,a2,Y)
        w1,b1,w2,b2=upd_params(w1,w2,b1,b2,dw1,dw2,db1,db2,alpha)
        '''if i%50==0:
            print("iteration",i)
            print("accuracy=", 100*accu(get_pred(a2),Y))'''
    return w1,b1,w2,b2

w1,b1,w2,b2=grad_des(X_train,Y_train,100,0.1)
z1,a1,z2,a2=for_prop(w1,b1,w2,b2,X_test)
print(get_pred(a2))

