#importing libraies

import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

#reading datas from csv file

data=pd.read_csv("D:\csv\car_data1.csv")

#x is input

#y is output

x=data.iloc[:, [2,3]].values
y=data.iloc[:, 4].values

#Data Splitting-training and test data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#scaling

st_x= StandardScaler()

x_train=st_x.fit_transform(x_train)

x_test= st_x.transform(x_test)

#Decison tree Model and training

classifer=SVC(kernel='linear', random_state=0)
classifer.fit(x_train,y_train)

#Prediction

y_pred=classifer.predict(x_test)

print("predicted out put")

print(y_pred)

print("\n")

print("actual output")

print(y_test)

print("\n")

#comparing actual and predicted output

#TP,TN,FP,FN

cm=confusion_matrix(y_test,y_pred)

print("confusion matrix \n ")

print(cm)
