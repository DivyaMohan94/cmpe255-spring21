import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

labels = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'] 

dataset = pd.read_csv('Data/housing.csv', header=None, delim_whitespace=True, names=labels)

RMSE=[]
RSquared = []

polynomial_regression_model = LinearRegression()

for i in range(len(labels)-1):
    X = dataset.iloc[:, i:i+1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    poly_reg = PolynomialFeatures(degree = 2)
    X_poly = poly_reg.fit_transform(X_train)
    polynomial_regression_model.fit(X_poly, y_train)
    y_pred = polynomial_regression_model.predict(poly_reg.fit_transform(X_test))
    RMSE.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    RSquared.append(metrics.r2_score(y_test, y_pred))

X = dataset.iloc[:, 12:13].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
polynomial_regression_model.fit(X_poly, y_train)
y_pred = polynomial_regression_model.predict(poly_reg.fit_transform(X_test))

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('House Price vs LSTAT')
plt.xlabel('LSTAT')
plt.ylabel('House Price')
plt.show()

# Result for 20 degrees
X = dataset.iloc[:, 12:13].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
poly_reg = PolynomialFeatures(degree = 20)
X_poly = poly_reg.fit_transform(X_train)
polynomial_regression_model.fit(X_poly, y_train)
y_pred = polynomial_regression_model.predict(poly_reg.fit_transform(X_test))

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('House Price vs LSTAT')
plt.xlabel('LSTAT')
plt.ylabel('House Price')
plt.show()