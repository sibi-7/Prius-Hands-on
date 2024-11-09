import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#Read data from file
data_set=pd.read_csv("D:\csv\car_data1.csv")
#x-input, y-output
x=data_set.iloc[:, [2,3]].values
y=data_set.iloc[:, 4].values
#Data splitting-training and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#Scaling data
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)
#Decison Tree model and training
classifier=RandomForestClassifier(n_estimators=10, criterion="entropy")
classifier.fit(x_train,y_train)
#Prediciton
y_pred=classifier.predict(x_test)
#comparing actual and predicted output #TP,TN,FP,FN
cm=confusion_matrix(y_test,y_pred)
print("confusion matrix \n")
print(cm)
