import numpy as np
import pandas as pd

dataset=pd.read_csv('auto_mpg_dataset.csv')

dataset=dataset.drop(['car_name'],axis='columns')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

print(model.score(x_test,y_test))