#linear regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Line 1
m0 = 1
b0 = 0

# Line 2
m1 = 3
b1 = 2

#  Line 3
m2 = -1/2
b2 = 2

xvals = np.linspace(-4,4)  # These are the x value for plotting

y0 = m0*xvals + b0

y1 = m1*xvals + b1

y2 = m2*xvals + b2

plt.figure(figsize = (15,10))

plt.axhline(y=0, color='black')
plt.axvline(x=0, color='black')

plt.grid(True, which='both')
plt.plot(xvals, y0, label = "$\ell_0$", color = "blue")

plt.plot(xvals, y1, label = "$\ell_1$", color = "red")

plt.plot(xvals, y2, label = "$\ell_2$", color = "green")


plt.xlim(-3,3)
plt.ylim(-3,3)


plt.legend()
plt.show()

Auto = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2025_DataRepository/main/Auto.csv')

Auto.head()
Auto.describe()

# Note we have no na's do deal with
Auto.isnull().sum(axis = 0)

Auto.plot.scatter('weight', 'mpg',figsize = (15,10))
plt.show()

from sklearn.linear_model import LinearRegression


X = Auto["weight"].values.reshape(-1, 1)
Y = Auto["mpg"].values.reshape(-1, 1)
linear_regressor1 = LinearRegression()
linear_regressor1.fit(X, Y)  # This finds the estimated coeficients.  The slope and intercepts.

print("The estimated intercept is ", linear_regressor1.intercept_[0])
print("The estimated slope is ", linear_regressor1.coef_[0,0])

Y_pred1 = linear_regressor1.predict(X)  ## This predicts the mpg given the weight of each car.

plt.figure(figsize = (15,10))


plt.scatter(X, Y)
plt.plot(X, Y_pred1, color='red', label="fit line")
plt.xlabel("weight")
plt.ylabel("mpg")

plt.legend()
plt.show()

#multi-linear regression

Auto.columns  # I am just reminding myself the column names.

predictors = ['cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration']   # These will play the roles of X1, X2, etc

X2 = Auto[predictors].values
Y = Auto["mpg"].values.reshape(-1, 1)
linear_regressor2 = LinearRegression()
linear_regressor2.fit(X2, Y)  # This estimates the coefficients

Y_pred2 = linear_regressor2.predict(X2)  # This predicts the mpg given the other information

# Model 1


model0_error = (Y.mean() - Y)

model1_error = (Y_pred1 - Y)


SSE0 = np.sum(model0_error**2)
SSE1 = np.sum(model1_error**2)

print("The R^2 for model 1 is ", 1 - SSE1/SSE0)

# Model 2


model2_error = (Y_pred2 - Y)
SSE2 = np.sum(model2_error**2)

print("The R^2 for model 2 is ", 1 - SSE2/SSE0)

#logistic regression
penguins = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2025_DataRepository/main/penguins.csv')

penguins.head()

penguins2 = pd.get_dummies(penguins, drop_first=True, dtype='int')  # drop_first=True, means that we don't have a redudant columns. Try setting it to FALSE and see.
penguins2

predictors = penguins2.columns[0:-1] # Don't include the last column.

X = penguins2[predictors].values  #
Y = penguins2["sex_male"].values.reshape(-1, 1)
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.figure(figsize = (15,10))

plt.scatter(Y_pred, Y)
plt.xlabel("Predited")
plt.ylabel("Actual")

plt.show()

xvals = np.linspace(-6,6)
f = lambda x: 1/(1+np.exp(-x))

plt.figure(figsize= (15,10))

plt.plot(xvals, f(xvals))

plt.axhline(0, color = "red")
plt.axhline(1, color = "red")

plt.show()

from sklearn.linear_model import LogisticRegression

logistic_regressor = LogisticRegression(max_iter = 10000)
logistic_regressor.fit(X, Y.ravel())  # This fits the model

Y_pred_prob = logistic_regressor.predict_proba(X)
Y_pred_prob[0:10,:]

plt.figure(figsize = (15,10))

plt.scatter(Y_pred_prob[:,1], Y)
plt.xlabel("Predited")
plt.ylabel("Actual")

plt.show()

Y_pred= logistic_regressor.predict(X)

# How well did this work?
from sklearn.metrics import confusion_matrix

confusion_matrix(Y,Y_pred)
confusion_matrix(Y,Y_pred, normalize = 'all')

#soft-max
iris_df = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2025_DataRepository/main/iris.csv')

iris_df.head()
iris_df.to_csv("iris.csv")

labels, unique = pd.factorize(iris_df['species'])
labels

X = iris_df.iloc[:,0:4].values
y = labels

mc_lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')

mc_lr.fit(X, y)
mc_lr.predict_proba(X)[0:5,:]

preds = mc_lr.predict(X)
confusion_matrix(y,preds)
