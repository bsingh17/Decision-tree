import numpy as np
import pandas as pd

dataset=pd.read_csv('chess_king_rook_dataset.csv')

from sklearn.preprocessing import LabelEncoder
lbl_white_king=LabelEncoder()
dataset['white_king_file']=lbl_white_king.fit_transform(dataset['white_king_file'])
lbl_white_rook=LabelEncoder()
dataset['white_rook_file']=lbl_white_rook.fit_transform(dataset['white_rook_file'])
lbl_black_king=LabelEncoder()
dataset['black_king_file']=lbl_black_king.fit_transform(dataset['black_king_file'])
lbl_result=LabelEncoder()
dataset['result']=lbl_result.fit_transform(dataset['result'])

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=100)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)

print(model.score(x_test,y_test))
