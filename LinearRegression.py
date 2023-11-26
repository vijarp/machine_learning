import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
companies = pd.read_csv("1000_Companies.csv")

first_row = list(companies.columns)
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


print("*"*25 + "Ridge"+"*"*25)
from sklearn.linear_model import Ridge

alphas = [0.1,1,10,100,1000,1000]
ridge_scores = []
for alpha in alphas:
    #Create a Ridge
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_train,y_train)
    score = ridge.score(X_test,y_test)
    ridge_scores.append(score)

print(ridge_scores)


print("*"*25 + " Lasso " + "*"*25)

from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.3)

lasso.fit(X,y)
lasso_coef = lasso.coef_
print(lasso_coef)
print(first_row)
plt.bar(first_row,lasso_coef)
plt.xticks(rotation = 45)
plt.show()

from sklearn.model_selection import Rand