import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('monk_dataset.csv')

from sklearn.preprocessing import LabelEncoder
lbl_problem=LabelEncoder()
dataset['problem']=lbl_problem.fit_transform(dataset['problem'])

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix 
confusion=confusion_matrix(y_test,y_predict)
print(confusion)

print(model.score(x_test,y_test))

plt.figure(figsize=(10,7))
plt.xlabel('Range')
plt.ylabel('class')
plt.scatter(range(0,428),y_predict,c='green')
plt.show()