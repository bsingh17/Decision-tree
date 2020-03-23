import numpy as np
import pandas as pd

data=pd.read_csv('computer_hardware_dataset.csv')
from sklearn.preprocessing import LabelEncoder
lbl_vendor=LabelEncoder()
data['vendor_name']=lbl_vendor.fit_transform(data['vendor_name'])
lbl_model=LabelEncoder()
data['model_name']=lbl_model.fit_transform(data['model_name'])

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=100)

from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)

print(reg.score(x_test,y_test))
