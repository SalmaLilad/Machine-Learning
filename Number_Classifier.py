import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import cluster
from sklearn.model_selection import train_test_split

mnist_df = pd.read_csv('https://raw.githubusercontent.com/sziccardi/MLCamp2024_DataRepository/main/mnist_train1.csv')

i = 100

sample = np.array(mnist_df.iloc[i,1:]);
sample = sample.reshape((28,28));
plt.imshow(sample, cmap='gray', interpolation='none')
plt.title(" Label: {}".format(mnist_df['label'].iloc[i].item()))
plt.xticks([])
plt.yticks([])
plt.show()

mnist_X = mnist_df.iloc[:,1:].values
mnist_y = mnist_df["label"].values
mnist_X_train, mnist_X_test, mnist_y_train, mnist_y_test = train_test_split(mnist_X, mnist_y, test_size=0.2, random_state=0)

mnist_classifier = KNeighborsClassifier(n_neighbors=10)
mnist_classifier.fit(mnist_X_train, mnist_y_train)

mnist_y_test_pred = mnist_classifier.predict(mnist_X_test)

import random
j = random.randint(0,mnist_y_test_pred.shape[0])

sample = mnist_X_test[j]
sample = sample.reshape((28,28))
plt.imshow(sample, cmap='gray', interpolation='none')
drawColor = 'black'
if mnist_y_test_pred[j].item() != mnist_y_test[j]:
    drawColor = 'red'
plt.title(" Label: {} Pred: {} ".format(mnist_y_test[j].item(), mnist_y_test_pred[j].item()), color=drawColor)
plt.xticks([])
plt.yticks([])
plt.show()

incor_mask = (mnist_y_test == mnist_y_test_pred)
incor_i = np.where(incor_mask == 0)[0]

k = incor_i[random.randint(0,incor_i.shape[0])]

sample = mnist_X_test[k]
sample = sample.reshape((28,28))
plt.imshow(sample, cmap='gray', interpolation='none')
drawColor = 'black'
if mnist_y_test_pred[k].item() != mnist_y_test[k]:
    drawColor = 'red'
plt.title(" Label: {} Pred: {} ".format(mnist_y_test[k].item(), mnist_y_test_pred[k].item()), color=drawColor)
plt.xticks([])
plt.yticks([])
plt.show()
