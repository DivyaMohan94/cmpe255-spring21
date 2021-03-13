import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

labels = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'] 

dataset = pd.read_csv('Data/housing.csv', header=None, delim_whitespace=True, names=labels)

RMSE, RSquared = [], []

#print(dataset.head(20))

linear_regression_model = LinearRegression()

for i in range(len(labels)-1):
    X = dataset.iloc[:, i:i+1].values
    Y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    linear_regression_model.fit(X_train, y_train)
    y_pred = linear_regression_model.predict(X_test)
    RMSE.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    RSquared.append(metrics.r2_score(y_test, y_pred))

RMSE_LSAT = min(RMSE)
RSQ_LSAT = max(RSquared)

print('RMSE of LSAT: ',RMSE_LSAT)
print('R squared: ',RSQ_LSAT)

X = dataset.iloc[:, 12:13].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

linear_regression_model.fit(X_train, y_train)

y_pred = linear_regression_model.predict(X_test)

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('LSTAT vs House Price')
plt.xlabel('LSTAT')
plt.ylabel('House Price')
plt.show()