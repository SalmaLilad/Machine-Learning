import pandas as pd

url = "https://raw.githubusercontent.com/charleslambert98/nba_combine_draft_data/main/nba_draft_combine_all_years.csv"

dataset = pd.read_csv(url)

dataset = dataset[['Draft pick', 'Standing reach', 'Wingspan','Vertical (Max)', 'Bench', 'Agility', 'Sprint']]
print(dataset.isnull().sum())

import matplotlib.pyplot as pyplot

dataset.hist(figsize = (20,15))
pyplot.show()

# Replace any missing values with 0
dataset = dataset.fillna(value={'Draft pick': 0})
dataset['Draft pick'] = dataset['Draft pick'].astype('int')
# Remove any rows with missing data
dataset = dataset.dropna()

print(dataset.isnull().sum())

import seaborn as sns

sns.pairplot(dataset, height = 1.5, aspect = 2.0)
pyplot.show()

from sklearn.model_selection import train_test_split

features = ['Wingspan', 'Vertical (Max)','Bench','Agility']
X = dataset[features]
y = dataset['Draft pick']

X_train, X_validation, y_train, y_validation = train_test_split(X, y)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
predictions = lda_model.predict(X_validation)
print(accuracy_score(y_validation, predictions))

from sklearn.naive_bayes import GaussianNB

gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)

predictions = gnb_model.predict(X_validation)
print(accuracy_score(y_validation, predictions))
