import numpy as np
import pandas as pd

dataset=pd.read_csv('survival rate on titanic ship.csv')

import math
age_median=math.floor(dataset.Age.median())
dataset.Age=dataset.Age.fillna(int(age_median))

inputs=dataset[['Pclass','Sex','Age','Fare']]
target=dataset['Survived']

from sklearn.preprocessing import LabelEncoder
lbl_sex=LabelEncoder()
inputs['Sex']=lbl_sex.fit_transform(inputs['Sex'])

from sklearn.model_selection import train_test_split
inputs_train,inputs_test,target_train,target_test=train_test_split(inputs,target,test_size=0.5,random_state=0)

from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(inputs_train,target_train)

targets_predict=model.predict(inputs_test)
print(model.score(inputs_train,target_train))