import numpy as np
import pandas as pd

dataset=pd.read_csv('season vs playing.csv')

from sklearn.preprocessing import LabelEncoder
lbl_outlook=LabelEncoder()
lbl_temprature=LabelEncoder()
lbl_humidity=LabelEncoder()
lbl_wind=LabelEncoder()
lbl_play=LabelEncoder()
dataset['Outlook']=lbl_outlook.fit_transform(dataset['Outlook'])
dataset['Temperature']=lbl_temprature.fit_transform(dataset['Temperature'])
dataset['Humidity']=lbl_humidity.fit_transform(dataset['Humidity'])
dataset['Wind']=lbl_wind.fit_transform(dataset['Wind'])
dataset['Play Tennis']=lbl_play.fit_transform(dataset['Play Tennis'])

inputs=dataset.drop('Play Tennis',axis='columns')
target=dataset['Play Tennis']



from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(inputs,target)
print(model.predict([[0,0,0,1]]))


