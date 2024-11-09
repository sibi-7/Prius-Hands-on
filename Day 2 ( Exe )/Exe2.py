# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Read data from file
data_set = pd.read_csv("D:\csv\car_data.csv")

# x-input, y-output
x = data_set.iloc[:, [2, 3]].values
y = data_set.iloc[:, 4].values

# Data splitting - training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Scaling data
x_st = StandardScaler()
x_trainst = x_st.fit_transform(x_train)
x_testst = x_st.transform(x_test)

# ML - Logistic Regression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_trainst, y_train)

# Prediction
y_pred = classifier.predict(x_test)
print("Predicted output:", y_pred)
print("Expected output:", y_test)

# Comparing actual and predicted output
# TP, TN, FP, FN
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix is : \n", cm)

y_pred1 = classifier.predict([[54, 60000]])
print("Predicted output:", y_pred1)
