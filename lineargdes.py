import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('Salary_dataset.csv')
data = data.iloc[:, 1:]



def sqloss(m,c,dpoints):
    error=0
    for i in range(len(dpoints)):
        x=dpoints.iloc[i].YearsExperience
        y=dpoints.iloc[i].Salary
        error+= (y-(m*x+c))**2
    return error/float(len(dpoints))

def graddes(m_now,c_now,dpoints,lr):
    m_grad=0
    c_grad=0
    n=len(dpoints)
    for i in range(n):
        x=dpoints.iloc[i].YearsExperience
        y=dpoints.iloc[i].Salary
        m_grad+= -(2/n)*x*(y-(m_now*x+c_now))
        c_grad+= -(2/n)*(y-(m_now*x+c_now))
    m=m_now-lr*m_grad
    c=c_now-lr*c_grad
    return m,c

m=0
c=0
lr=0.01
epochs=100


for i in range(epochs):
    m,c=graddes(m,c,data,lr)

loss=sqloss(m,c,data)

print(f"the final values of m and c and loss are {m} and {c} and {loss}")
plt.plot(list(range(0,13)),[m*x+c for x in range(0,13)],color="red")
plt.scatter(data.YearsExperience,data.Salary,color="black")
plt.show()
