import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris_df = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2025_DataRepository/main/iris.csv')

iris_df.head()
iris_df.shape

iris_df['species'].value_counts()
iris_df.describe()

plt.figure()
sns.pairplot(iris_df, vars=iris_df.columns[1:-1], hue = "species")
plt.show()

#k-nearest neighbor
feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X = iris_df[feature_names].values
y = iris_df["species"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

#finding k optimal value
k_list = list(range(1,50,2))
cv_scores = []
k_list

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv = StratifiedKFold(n_splits=10, shuffle=True), scoring = 'accuracy')
    cv_scores.append(scores.mean())

plt.figure()
plt.title('Performance of K Nearest Neighbors Algorithm')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy Score')
plt.plot(k_list, cv_scores)

plt.show()

best_k = k_list[cv_scores.index(max(cv_scores))]
print(best_k)
