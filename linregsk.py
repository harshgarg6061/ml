import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df=pd.read_csv("Salary_dataset.csv")
#print(df.head())
X=np.array(df['YearsExperience']).reshape(-1,1)
Y=np.array(df['Salary']).reshape(-1,1)
'''x_train,x_test,y_train,y_test=train_test_split(X,Y)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)'''
model=LinearRegression()
model.fit(X,Y)
y_pred=model.predict(X)
print(y_pred)


'''plt.scatter(y_test,y_pred)
plt.axline((0, 0), slope=1)
plt.show()'''
