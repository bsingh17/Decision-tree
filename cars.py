import numpy as np
import pandas as pd

dataset=pd.read_csv('cars_dataset.csv')

from sklearn.preprocessing import LabelEncoder
lbl_buying=LabelEncoder()
dataset['buying']=lbl_buying.fit_transform(dataset['buying'])
lbl_maint=LabelEncoder()
dataset['maint']=lbl_maint.fit_transform(dataset['maint'])
lbl_doors=LabelEncoder()
dataset['doors']=lbl_doors.fit_transform(dataset['doors'])
lbl_persons=LabelEncoder()
dataset['persons']=lbl_persons.fit_transform(dataset['persons'])
lbl_lugboot=LabelEncoder()
dataset['lug_boot']=lbl_lugboot.fit_transform(dataset['lug_boot'])
lbl_safety=LabelEncoder()
dataset['safety']=lbl_safety.fit_transform(dataset['safety'])
lbl_car=LabelEncoder()
dataset['car']=lbl_car.fit_transform(dataset['car'])

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.tree import DecisionTreeClassifier
reg=DecisionTreeClassifier()
reg.fit(x_train,y_train)

y_predict=reg.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)
print(reg.score(x_test,y_test))