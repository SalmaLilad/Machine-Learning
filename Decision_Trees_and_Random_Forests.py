import pandas as pd

bc_data = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2025_DataRepository/main/breast_cancer.csv')

bc_data.head()
bc_data.columns
bc_data.shape

bc_data['diagnosis'].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.pairplot(bc_data, vars=['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean'], hue = "diagnosis")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

x = bc_data[feature_names].values
y = bc_data['diagnosis'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

confusion_matrix(y_test, y_pred)
accuracy_score(y_pred, y_test)

y_pred = clf.predict(x_train)
accuracy_score(y_pred, y_train)

import graphviz

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph

print(feature_names[7])
print(feature_names[20])
print(feature_names[21])

#random forests
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf = rf.fit(x_train, y_train)

rf_pred = rf.predict(x_test)
accuracy_score(rf_pred, y_test)
