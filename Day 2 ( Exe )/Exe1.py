import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("D:\csv\car_data1.csv")
print(data.head())

# independent variables for linear regression
x = data[['Weight', 'Volume']]
print(x)

# dependent variables for linear regression
y = data['CO2']
print(y)

plt.scatter(data['Weight'], data['CO2'], s=15*np.pi)
a1, b1 = np.polyfit(data['Weight'], data['CO2'], 1)
plt.plot(data['Weight'], a1*data['Weight'] + b1, 'r')
plt.xlabel('Weight')
plt.ylabel('CO2')
plt.title("CO2 vs Weight")
plt.show()

plt.scatter(data['Volume'], data['CO2'], s=15*np.pi)
a2, b2 = np.polyfit(data['Volume'], data['CO2'], 1)
plt.plot(data['Volume'], a2*data['Volume'] + b2, 'r')
plt.xlabel('Volume')
plt.ylabel('CO2')
plt.title("CO2 vs Volume")
plt.show()

# Linear regression model
# Data splitting - training and test data
x = data[['Weight', 'Volume']]
y = data['CO2']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

linear_regr = linear_model.LinearRegression()
# training the model
linear_regr.fit(x_train, y_train)
# model output - prediction of CO2 based on weight and volume
y_pred = linear_regr.predict(x_test)
predict_CO2 = linear_regr.predict([[2000, 1400]])
print(predict_CO2)
print("*Performance Metrics*")
mse = mean_squared_error(y_test, y_pred)
print(mse)
r2_score=r2_score(y_test,y_pred)
print(f"Mean squared error is:{mse:.2f}")
print(f"r2 score :{r2_score:.2f}")
