import pandas as pd
music_df = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2024_DataRepository/main/music.csv')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

x = music_df.iloc[:, 1:-1]
x.head()
x.describe()

x=x.dropna()
x.describe()
music_df.columns

X_value = x.values
Y_value = music_df['label'].values
selected_features = music_df[['chroma_stft'] + ['mfcc' + str(i) for i in range(1, 21)]].values
X_train, X_test, y_train, y_test = train_test_split(selected_features, Y_value, test_size=0.2, random_state=0)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_train_pred = classifier.predict(X_train)

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
accuracy = accuracy_score(y_train, y_train_pred)
print(accuracy)

from sklearn import svm
svc = svm.SVC(kernel='linear')
svc = svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
y_train_pred = svc.predict(X_train)
confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
accuracy = accuracy_score(y_train, y_train_pred)
print(accuracy)
