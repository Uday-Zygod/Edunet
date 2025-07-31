import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/water_potability.csv')
df.head()
x=df.drop(columns=["Potability"])
x
y=df["Potability"]
y
df.info()
df.isnull().sum()
df["ph"].fillna(df["ph"].mean(),inplace=True)
df["Sulfate"].fillna(df["Sulfate"].mean(),inplace=True)
df["Trihalomethanes"].fillna(df["Trihalomethanes"].mean(),inplace=True)
df.isnull().sum()
x=df.drop(columns=["Potability"])
y=df["Potability"]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)
from sklearn.metrics import classification_report
ypred=model.predict(xtest)
print(classification_report(ytest,ypred))
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)
print(classification_report(ytest,ypred))
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)
print(classification_report(ytest,ypred))
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)
print(classification_report(ytest,ypred))
