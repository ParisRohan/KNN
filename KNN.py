import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

dataset=pd.read_csv("sample.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,2].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

Xnew_test=np.array([158,60])
y_predict=knn.predict([Xnew_test])
print("Prediction = ",y_predict)

print("Prediction on x test data =")
y_pred=knn.predict(x_test)
print(y_pred)

print(confusion_matrix(y_test,y_pred))

print("Accuracy score= ",accuracy_score(y_test,y_pred)*100)


