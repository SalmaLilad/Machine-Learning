import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import cluster
from sklearn.model_selection import train_test_split

baseball_df = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2024_DataRepository/main/baseball.csv')
print("Data shape before cleaning:", baseball_df.shape)

baseball_df.describe()

total_baseball_cols = baseball_df.columns
num_baseball_cols = baseball_df._get_numeric_data().columns # list of columns that ARE numeric data
print("Numerical columns: ", num_baseball_cols)

cat_baseball_cols = list(set(total_baseball_cols) - set(num_baseball_cols)) # list of columns that are NOT numeric data (categorical data)
print("Categorical columns: ", cat_baseball_cols)

baseball_df[cat_baseball_cols].describe()

print(baseball_df['pitch_name'].unique())
print(baseball_df['pitch_type'].unique())
print(baseball_df['events'].unique())
print(baseball_df['description'].unique())
print(baseball_df['stand'].unique())
print(baseball_df['pitch_name'].unique())
print(baseball_df['pitch_type'].unique()) # abbreviations

baseball_df['pitch_name'].value_counts()

clean_baseball_df = baseball_df.dropna()
print("Shape after cleaning: ", clean_baseball_df.shape)

feature_names = ['release_speed', 'release_pos_x', 'release_pos_z', 'batter', 'pitcher',
       'zone', 'balls', 'strikes', 'outs_when_up', 'inning', 'hit_distance_sc',
       'launch_speed', 'launch_angle', 'effective_speed', 'release_spin_rate',
       'release_extension', 'release_pos_y', 'pitch_number', 'home_score',
       'away_score', 'bat_score', 'fld_score'] # Ask yourself: are these all good features to predict the pitch type?

X_bb = clean_baseball_df[feature_names].values
y_bb = clean_baseball_df["pitch_name"].values

X_bb_train, X_bb_test, y_bb_train, y_bb_test = train_test_split(X_bb, y_bb, test_size=0.2, random_state=0)

pitch_classifier = KNeighborsClassifier(n_neighbors=5)
pitch_classifier.fit(X_bb_train, y_bb_train)

y_bb_test_pred = pitch_classifier.predict(X_bb_test) # predicting on unseen data
bb_test_accuracy = accuracy_score(y_bb_test, y_bb_test_pred)
print("Test accuracy:", bb_test_accuracy)
y_bb_train_pred = pitch_classifier.predict(X_bb_train) # predicting on seen data
bb_train_accuracy = accuracy_score(y_bb_train, y_bb_train_pred)
print("train: ", bb_train_accuracy)

#Those were not great results -> focus on a smaller, similar problem to get better results
clean_baseball_subset_df = clean_baseball_df[ (clean_baseball_df['pitch_name'] == '2-Seam Fastball') |
                                              (clean_baseball_df['pitch_name'] == 'Slider') |
                                              (clean_baseball_df['pitch_name'] == 'Changeup')]
print("Subset shape:", clean_baseball_subset_df.shape)
clean_baseball_subset_df['pitch_name'].value_counts()

X_bb = clean_baseball_subset_df[feature_names].values # HINT: use your baseball knowledge and data visualization techniques to reduce the features!
y_bb = clean_baseball_subset_df["pitch_name"].values

X_bb_train, X_bb_test, y_bb_train, y_bb_test = train_test_split(X_bb, y_bb, test_size=0.2, random_state=0)

pitch_classifier = KNeighborsClassifier(n_neighbors=5)
pitch_classifier.fit(X_bb_train, y_bb_train)

y_bb_test_pred = pitch_classifier.predict(X_bb_test) # predicting on unseen data
bb_test_accuracy = accuracy_score(y_bb_test, y_bb_test_pred)
print("Test accuracy:", bb_test_accuracy)
y_bb_train_pred = pitch_classifier.predict(X_bb_train) # predicting on seen data
bb_train_accuracy = accuracy_score(y_bb_train, y_bb_train_pred)
print("train: ", bb_train_accuracy)
