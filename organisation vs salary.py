import pandas as pd

dataset=pd.read_csv('organisation vs salary.csv')
inputs=dataset.drop('salary_more_then_100k',axis='columns')
target=dataset['salary_more_then_100k']

from sklearn.preprocessing import LabelEncoder
lbl_company=LabelEncoder()
lbl_job=LabelEncoder()
lbl_degree=LabelEncoder()
inputs['company']=lbl_company.fit_transform(dataset['company'])
inputs['job']=lbl_job.fit_transform(dataset['job'])
inputs['degree']=lbl_degree.fit_transform(dataset['degree'])

from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(inputs,target)

print(model.score(inputs,target))
y_predict=model.predict(inputs)