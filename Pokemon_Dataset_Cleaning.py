import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import cluster
from sklearn.model_selection import train_test_split

poke_df = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2024_DataRepository/main/pokemon.csv')

poke_df.shape
total_poke_cols = poke_df.columns
num_cols = poke_df._get_numeric_data().columns
print("numerical columns: ", num_cols)

cat_cols = list(set(total_poke_cols) - set(num_cols))
print("categorical columns: ", cat_cols)
poke_df[cat_cols].describe()

cols_to_rm = ['name', 'abilities', 'classfication', 'japanese_name', 'is_legendary']
cols_to_dummy = list(set(cat_cols) - set(cols_to_rm))
print("columns to make dummy variables of: ", cols_to_dummy)

poke_df_dum = pd.get_dummies(poke_df, columns=cols_to_dummy)
print("shape after dummying: ", poke_df_dum.shape)

drop_dum_poke_df = poke_df_dum.dropna()
print("shape after dropping: ", drop_dum_poke_df.shape)

poke_cols = drop_dum_poke_df.columns
print(poke_cols)

poke_cols = poke_cols.drop(cols_to_rm)

poke_X = drop_dum_poke_df[poke_cols].values
poke_y = drop_dum_poke_df["is_legendary"].values
poke_X_train, poke_X_test, poke_y_train, poke_y_test = train_test_split(poke_X, poke_y, test_size=0.2, random_state=0)

poke_classifier = KNeighborsClassifier(n_neighbors=10)
poke_classifier.fit(poke_X_train, poke_y_train)

poke_y_test_pred = poke_classifier.predict(poke_X_test)
poke_test_accuracy = accuracy_score(poke_y_test, poke_y_test_pred)
print("test: ", poke_test_accuracy)
poke_y_train_pred = poke_classifier.predict(poke_X_train)
poke_train_accuracy = accuracy_score(poke_y_train, poke_y_train_pred)
print("train: ", poke_train_accuracy)

#Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Get confusion matrix
cm = confusion_matrix(poke_y_test, poke_y_test_pred, labels=[0, 1])

# Plot it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Legendary", "Legendary"])
disp.plot(cmap="Greens", values_format="d")
plt.title("Confusion Matrix: KNN Pokémon Classifier (k=10)")
plt.tight_layout()
plt.show()




