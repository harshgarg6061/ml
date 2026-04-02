import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('Salary_dataset.csv')
data = data.iloc[:, 1:]

class linearregclosed:

    def __init__(self):
        self.A=None

    def fit(self,X,Y):
        X=np.array(X)
        Y=np.array(Y)
        Xb=np.c_[np.ones((X.shape[0],1)),X]
        A=np.linalg.inv((Xb.T)@Xb)@Xb.T@Y
        self.A=A

    def predict(self,X):
        X=np.array(X)
        Xb=np.c_[np.ones((X.shape[0],1)),X]
        Yp=Xb@self.A
        return Yp
        
egample=linearregclosed()
X=data['YearsExperience']
Y=data['Salary']
egample.fit(X,Y)
print(egample.predict(X))