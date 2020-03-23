import numpy as np
import pandas as pd

dataset=pd.read_csv('forbes_billionaire_dataset.csv')
dataset=dataset.drop('name',axis='columns')
from sklearn.preprocessing import LabelEncoder
lbl_source=LabelEncoder()
dataset['source']=lbl_source.fit_transform(dataset['source'])
lbl_country=LabelEncoder()
dataset['country_citizenship']=lbl_country.fit_transform(dataset['country_citizenship'])

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=100)

from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)

print(reg.score(x_test,y_test))