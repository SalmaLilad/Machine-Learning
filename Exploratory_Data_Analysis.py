import pandas as pd

loan_df = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2025_DataRepository/main/credit.csv')

loan_df.head()
loan_df.columns

loan_df.describe(include='all')

loan_df.dtypes

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.pairplot(loan_df, vars=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History'], hue = "Loan_Status")
plt.show()

loan_df['Gender'].value_counts()

men_df = loan_df[loan_df['Gender'] == 'Male']
women_df = loan_df[loan_df['Gender'] == 'Female']

men_df.describe(include='all')
women_df.describe(include='all')

men_df['Loan_Status'].value_counts()
women_df['Loan_Status'].value_counts()

#Exercises
loan_df.loc[:,["Loan_Status", "Married"]].value_counts()

dropped_loan_df = loan_df.dropna()
dropped_loan_df.describe(include='all')

loan_df['Self_Employed'].value_counts()

loan_df_new = pd.get_dummies(loan_df, columns=['Self_Employed'])
loan_df_new.head()

loan_df.info()

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Gender', data=loan_df)
plt.show()
