import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import joblib

df = pd.read_csv("train.csv")
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())
df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
df['Married'] = df["Married"].fillna(df['Married'].mode()[0])
df['Dependents'] = df["Dependents"].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df["Self_Employed"].fillna(df['Self_Employed'].mode()[0])

df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Total_Income_Log'] = np.log(df['Total_Income'])
df['LoanAmountLog'] = np.log(df['LoanAmount'])

from sklearn.preprocessing import LabelEncoder
cols = ['Gender',"Married","Education",'Self_Employed',"Property_Area","Loan_Status","Dependents"]
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

cols=['Loan_ID','ApplicantIncome','CoapplicantIncome','LoanAmount','Total_Income']
df = df.drop(columns=cols , axis =1)
x = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(x, y , test_size = 0.1 , random_state = 0)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
names = []
scores = []

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state = 0)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
from sklearn import metrics
print("The accuracy of Random Forest is : " ,metrics.accuracy_score(y_pred , y_test)*100,'%')
names.append('RandomForestClassifier')
scores.append(metrics.accuracy_score(y_pred , y_test)*100)

from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train , y_train)
y_pred = NB.predict(X_test)
from sklearn import metrics
print("The accuracy of Naive Bayes is : " ,metrics.accuracy_score(y_pred , y_test)*100,'%')
names.append('GaussianNB')
scores.append(metrics.accuracy_score(y_pred , y_test)*100)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
from sklearn import metrics
print("The accuracy of Logistic Regression is : " ,metrics.accuracy_score(y_pred , y_test)*100,'%')
names.append('LogisticRegression')
scores.append(metrics.accuracy_score(y_pred , y_test)*100)

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC.fit(X_train, y_train)
y_pred = DTC.predict(X_test)
from sklearn import metrics
print("The accuracy of Decision Tree Classifier is : " ,metrics.accuracy_score(y_pred , y_test)*100,'%')
names.append('DecisionTreeClassifier')
scores.append(metrics.accuracy_score(y_pred , y_test)*100)

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train , y_train)
y_pred = KNN.predict(X_test)
from sklearn import metrics
print("The accuracy of K Nearest Neighbors is : " ,metrics.accuracy_score(y_pred , y_test)*100,'%')
names.append('KNeighborsClassifier')
scores.append(metrics.accuracy_score(y_pred , y_test)*100)

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train , y_train)
y_pred = gbc.predict(X_test)
from sklearn import metrics
print("The accuracy of K GradientBoostingClassifier is : " ,metrics.accuracy_score(y_pred , y_test)*100,'%')
names.append('GradientBoostingClassifier')
scores.append(metrics.accuracy_score(y_pred , y_test)*100)

print(names)
print(scores)

pickle.dump(LR, open('mod.pkl', 'wb'))