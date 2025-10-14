import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

police_df = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2025_DataRepository/refs/heads/main/Police_stop_data1.csv')

police_df.head()
police_df.columns
police_df.shape

dropped_police_df = police_df.dropna()
dropped_police_df.describe(include='all')

police_df.dtypes
police_df.info()


police_df_new = pd.get_dummies(police_df, columns=['reason',
       'problem', 'callDisposition', 'citationIssued',
       'vehicleSearch', 'preRace', 'race', 'gender', 'personSearch'], drop_first=True, dtype = 'int')
police_df_new = police_df_new.dropna()
police_df_new.head()

police_df_new.columns

police_df_new['personSearch_YES'].value_counts()


sns.countplot(x='personSearch_YES', data=police_df_new)
plt.show()
sns.countplot(x='problem_Curfew Violations (P)', data=police_df_new)
plt.show()
sns.countplot(x='problem_Suspicious Person (P)', data=police_df_new)
plt.show()


race_black_df = police_df[police_df['preRace'] == 'White']

race_black_df.head()
race_black_df['personSearch'].value_counts()
sns.countplot(race_black_df, x='problem', hue='personSearch' )
plt.xticks(rotation=90)

raCE_BLACK_df = police_df[police_df['preRace'] == 'Asian']

raCE_BLACK_df.head()
palette = {'No': 'blue', 'Yes': 'orange'}
raCE_BLACK_df['personSearch'].value_counts()
sns.countplot(raCE_BLACK_df, x='problem', hue='personSearch' )
plt.xticks(rotation=90)

raCE_BLACK_df = police_df[police_df['preRace'] == 'Black']

raCE_BLACK_df.head()
raCE_BLACK_df['personSearch'].value_counts()
sns.countplot(raCE_BLACK_df, x='problem', hue='personSearch')
palette = {'No': 'blue', 'Yes': 'orange'}
plt.xticks(rotation=90)

problems = ['problem_Curfew Violations (P)',
       'problem_Suspicious Person (P)', 'problem_Suspicious Vehicle (P)',
       'problem_Traffic Law Enforcement (P)', 'problem_Truancy (P)']

ps = police_df_new['policePrecinct']
ps.hist(bins=50);

police_df['personSearch'].value_counts()


#Split your data into training and testing sets
feature_names = ['reason_Equipment Violation', 'reason_Investigative',
       'reason_Moving Violation','vehicleSearch_YES',
       'policePrecinct', 'personSearch_YES']

#X = police_df_new[feature_names].values
#y = police_df_new['personSearch_YES'].values

subset = police_df_new[feature_names]
predictors = subset.columns[0:-1] # Don't include the last column.

X = subset[predictors].values  #
y = subset["personSearch_YES"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#Manipulating Data

#knn
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred)
print(cm)
accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy)

#descision trees & forest classifier
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

clf_new = tree.DecisionTreeClassifier(max_depth=2)
clf_new = clf_new.fit(X_train, Y_train)

y_pred = clf_new.predict(X_test)

import graphviz

dot_data = tree.export_graphviz(clf_new, out_file=None)
graph = graphviz.Source(dot_data)
graph

confusion_matrix(Y_test, y_pred)
accuracy_score(y_pred, Y_test)

print(feature_names[3])
print(feature_names[1])
print(feature_names[4])

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf = rf.fit(X_train, Y_train)

rf_pred = rf.predict(X_test)
accuracy_score(rf_pred, Y_test)
rf_pred = rf.predict(X_train)
accuracy_score(rf_pred, y_train)

#SVM
#Train your svm classifier
from sklearn.model_selection import train_test_split

features = [
       'problem_Curfew Violations (P)', 'problem_Suspicious Person (P)',
       'problem_Suspicious Vehicle (P)', 'problem_Traffic Law Enforcement (P)',
       'problem_Truancy (P)']

loan_X = police_df_new[features]
loan_y = police_df_new['personSearch_YES']

loan_X_train, loan_X_test, loan_y_train, loan_y_test = train_test_split(loan_X, loan_y, test_size=0.25, random_state=42)

#Predict classes for the testing set
from sklearn import svm

svc = svm.SVC(kernel='linear')
svc = svc.fit(loan_X_train, loan_y_train)
y_pred = svc.predict(loan_X_test)
accuracy_score(y_pred, loan_y_test)

#Predict classes for the testing set
from sklearn import svm

svc = svm.SVC(kernel='rbf')
svc = svc.fit(loan_X_train, loan_y_train)
y_pred = svc.predict(loan_X_test)
accuracy_score(y_pred, loan_y_test)

#Predict classes for the testing set
from sklearn import svm

svc = svm.SVC(kernel='poly')
svc = svc.fit(loan_X_train, loan_y_train)
y_pred = svc.predict(loan_X_test)
accuracy_score(y_pred, loan_y_test)

#Predict classes for the testing set
from sklearn import svm

svc = svm.SVC(kernel='sigmoid')
svc = svc.fit(loan_X_train, loan_y_train)
y_pred = svc.predict(loan_X_test)
accuracy_score(y_pred, loan_y_test)

#Evaluate your model the first way
from sklearn.metrics import confusion_matrix
confusion_matrix(loan_y_test, y_pred)

#Evaluate your model the second way
from sklearn.metrics import accuracy_score
accuracy_score(y_pred, loan_y_test)

#Logistical Regression
police_df_new = pd.get_dummies(police_df, columns=['reason',
       'problem', 'callDisposition', 'citationIssued',
       'vehicleSearch', 'preRace', 'race', 'gender', 'personSearch'], drop_first=True, dtype = 'int')
police_df_new = police_df_new.dropna()
police_df_new.head()

#Split your data into training and testing sets
feature_names = ['reason_Equipment Violation', 'reason_Investigative',
       'reason_Moving Violation','vehicleSearch_YES',
       'policePrecinct', 'personSearch_YES']

#X = police_df_new[feature_names].values
#y = police_df_new['personSearch_YES'].values

subset = police_df_new[feature_names]
predictors = subset.columns[0:-1] # Don't include the last column.

X = subset[predictors].values  #
y = subset["personSearch_YES"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

logistic_regressor = LogisticRegression(max_iter = 10000)
logistic_regressor.fit(X_train, y_train.ravel())  # This fits the model

Y_pred_prob = logistic_regressor.predict_proba(X_test)
Y_pred_prob[0:10,:]

plt.figure(figsize = (15,10))


plt.scatter(Y_pred_prob[:,1], y_test, alpha=.01)
plt.xlabel("Predited")
plt.ylabel("Actual")
plt.show()

Y_pred= logistic_regressor.predict(X_test)
cm = confusion_matrix(y_test,Y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Searched", "Searched"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix: Was A Person Searched (LR)")
plt.tight_layout()
plt.show()

confusion_matrix(y_test,Y_pred, normalize = 'all')
accuracy_score(y_test, Y_pred)
