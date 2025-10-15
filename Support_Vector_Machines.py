import pandas as pd

loan_df = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2025_DataRepository/main/credit.csv')

loan_df.head()
loan_df.columns

loan_df.describe(include='all')

loan_df['Loan_Status'].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.pairplot(loan_df, vars=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History'], hue = "Loan_Status")
plt.show()

new_loan_df = loan_df.dropna()
new_loan_df.describe(include="all")

from sklearn.model_selection import train_test_split

features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History']

loan_X = new_loan_df[features]
loan_y = new_loan_df['Loan_Status']

loan_X_train, loan_X_test, loan_y_train, loan_y_test = train_test_split(loan_X, loan_y, test_size=0.25, random_state=42)

from sklearn import svm

svc = svm.SVC(kernel='linear')
svc = svc.fit(loan_X_train, loan_y_train)

y_pred = svc.predict(loan_X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

confusion_matrix(loan_y_test, y_pred)
accuracy_score(y_pred, loan_y_test)

#kernel trick
rbf_svc = svm.SVC(kernel='rbf')
rbf_svc = rbf_svc.fit(loan_X_train, loan_y_train)

y_pred = rbf_svc.predict(loan_X_test)
accuracy_score(y_pred, loan_y_test)

poly_svc = svm.SVC(kernel='poly')
poly_svc = poly_svc.fit(loan_X_train, loan_y_train)

y_pred = poly_svc.predict(loan_X_test)
accuracy_score(y_pred, loan_y_test)

sig_svc = svm.SVC(kernel='sigmoid')
sig_svc = sig_svc.fit(loan_X_train, loan_y_train)

y_pred = sig_svc.predict(loan_X_test)
accuracy_score(y_pred, loan_y_test)
