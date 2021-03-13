
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

labels = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

dataset = pd.read_csv('Data/housing.csv', header=None, delim_whitespace=True, names=labels)

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

multiple_regression_model = LinearRegression()
multiple_regression_model.fit(X_train, y_train)

y_pred = multiple_regression_model.predict(X_test)

Rmse_Val = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
RSqVal = metrics.r2_score(y_test, y_pred)

print('RMSE value - ',Rmse_Val)

print('R Squared value- ',RSqVal)

print('Adjusted R Squared value- ',1 - (1-metrics.r2_score(y_test, y_pred))*(len(y_test)-1)/(len(y_test)-X_train.shape[1]-1))
