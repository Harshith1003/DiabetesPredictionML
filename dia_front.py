import pandas as pd 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
sns.set()
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV 
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import precision_score, recall_score,f1_score 
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('diabetes.csv')
import numpy as np 
d_copy=df.copy(deep=True)
sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(d_copy.drop(["Outcome"], axis=1))

columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = pd.DataFrame(X_scaled, columns=columns)
y=d_copy.Outcome
X=df.drop('Outcome',axis=1) 
y=df[ 'Outcome']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

from sklearn.linear_model import LogisticRegression
Ir=LogisticRegression()
Ir.fit(X_train,y_train) 
from sklearn import metrics
predictions=Ir.predict(X_test)

import pickle
pickle.dump(Ir, open("Diabetes.pkl","wb")) 
pickle.dump(sc_X,open('scaler.pkl', "wb"))
model=pickle.load(open("Diabetes.pkl","rb"))