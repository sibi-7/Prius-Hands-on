#working code

#import libraries

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#load data set from skikit

iris=load_iris()

#x-input,y-output
x=iris.data
y=iris.target
print("\nx: ")
print(x)

print("\nx: ")
print(y)

#splitting x and y into training and testing sets

#40% data as test set and 60% as train data

X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.4,random_state=1)

#supervised learning-training the mode on training set

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#predict the output

y_pred=knn.predict(X_test)

pred_species= [iris.target_names[p] for p in y_pred]
print("\n Predicitions", pred_species)

#comparing actual response values (y_test) with predicted responce values (y_pred)

print("\n KNN model accuracy", metrics.accuracy_score(y_test,y_pred))

#making prediction for out of smaple data

sample=[[3,5,4,2],[2,3,5,4]]
preds=knn.predict(sample)
pred_species=[iris.target_names[p] for p in preds]
print("\n Predictions",pred_species)
