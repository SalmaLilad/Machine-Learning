import numpy as np
import pandas as pd

housing_data = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2025_DataRepository/main/housing.csv')

housing_data.shape
housing_data.columns

housing_data.head()
housing_data.describe()

housing_data.describe(include='all')

pop = housing_data['population']
pop

print("mean:")
print(pop.mean())
print("median:")
print(pop.median())
print("std:")
print(pop.std())

pop.hist(bins=50);
pop.hist(bins=10);
pop.hist(bins=1000);

prox = housing_data['ocean_proximity']
prox.value_counts()

selection = housing_data.loc[:,['longitude', 'latitude', 'median_income']]
selection.head()

selection2 = housing_data.iloc[:, 0:3]
selection2.head()

selection.plot.scatter(x='longitude', y='latitude');
selection.plot.scatter(x='latitude', y='median_income');

#Exercise
h_data = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2025_DataRepository/refs/heads/main/world-happiness-report-2021.csv')

h_data.shape
h_data.columns

h_data.head()
h_data.describe(include='all')

h_data['Ladder score'].hist(bins=5);
h_data['Ladder score'].hist(bins=10)
h_data['Ladder score'].hist(bins=20)

h_data2 = h_data.loc[:,['Ladder score', 'Healthy life expectancy', 'Freedom to make life choices']]
h_data2.head()

h_data2.plot.scatter(x='Healthy life expectancy', y='Ladder score');
