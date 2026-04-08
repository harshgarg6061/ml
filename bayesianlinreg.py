import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df=pd.read_csv("Salary_dataset.csv")

X=np.array(df['YearsExperience']).reshape(-1,1)
Y=np.array(df['Salary']).reshape(-1,1)

X = (X - X.mean()) / X.std()
Y = (Y - Y.mean()) / Y.std()

class baylinreg:
    def __init__(self,alpha,beta):
        self.alpha=alpha
        self.beta=beta

    def fit(self,X,Y):
        X=np.c_[np.ones((X.shape[0],1)),X]
        I=np.eye(X.shape[1])
        self.S_N=np.linalg.inv(self.alpha*I + self.beta*X.T@X)
        self.m_N=self.beta*self.S_N@X.T@Y

    def predict(self,X_new):
        X_new= np.c_[np.ones((X_new.shape[0],1)),X_new]
        Y_mean=X_new@self.m_N
        variance=1/self.beta + np.sum(X_new @ self.S_N * X_new, axis=1, keepdims=True)#most important step
        return Y_mean,variance
    
egample=baylinreg(1,1)
egample.fit(X,Y)
mean,variance=egample.predict(X)

#print("Posterior mean (weights):\n", egample.m_N)
#print("Predictions:\n", mean)
#print("Uncertainty (variance):\n", variance)

for i in [0, 10, 20]:

    x_point = X[i].reshape(1, -1)
    mean_i, var_i = egample.predict(x_point)

    mu = mean_i[0, 0]
    var = var_i[0, 0]
    sigma = np.sqrt(var)

    y_vals = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    pdf = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-(y_vals - mu)**2 / (2*var))

    plt.plot(y_vals, pdf, label=f"x={X[i][0]:.2f}")
    plt.axvline(mu, linestyle='--')
    plt.axvline(Y[i, 0], color='red', linestyle=':')

plt.xlabel("Salary")
plt.ylabel("Probability Density")
plt.title("Predictive Distributions")
plt.legend()

plt.show()