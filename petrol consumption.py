import numpy as np
import pandas as pd

dataset=pd.read_csv('petrol_consumption.csv')
inputs=dataset.drop('Petrol_Consumption',axis='columns')
target=dataset['Petrol_Consumption']

from sklearn.model_selection import train_test_split
inputs_train,inputs_test,target_train,target_test=train_test_split(inputs,target,test_size=0.3,random_state=0)

from sklearn import tree
model=tree.DecisionTreeRegressor()
model.fit(inputs_train,target_train)
target_predict=model.predict(inputs_test)

print(model.score(inputs_test,target_predict))