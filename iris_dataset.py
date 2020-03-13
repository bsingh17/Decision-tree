import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

df=load_iris()
dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names
dataset['TARGET']=df.target
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print(model.score(x_test,y_predict))
