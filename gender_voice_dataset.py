import numpy as np
import pandas as pd

dataset=pd.read_csv('gender_voice_dataset.csv')

from sklearn.preprocessing import LabelEncoder
lbl_label=LabelEncoder()
dataset['label']=lbl_label.fit_transform(dataset['label'])

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=10)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)

print(model.score(x_test,y_test))