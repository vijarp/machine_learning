import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
companies = pd.read_csv("1000_Companies.csv")
X = companies.iloc[:,:-1].values
y = companies.iloc[:,4].values
print(companies.head())
#labelencoder = LabelEncoder()
#X[:,3] = labelencoder.fit_transform(X[:,3])

ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)



#print(X[:4])

X = X[:, 1:]
#print(X[:4])
#print(X.shape)
#print(sns.heatmap(companies.corr()))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print(X_test[:5])
print(y_pred[:5])
print(regressor.coef_)
print(regressor.intercept_)

from sklearn.metrics import r2_score

print(r2_score(y_test,y_pred))
