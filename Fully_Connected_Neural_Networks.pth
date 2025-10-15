import numpy as np
import matplotlib.pyplot as plt

t = np.arange(-10,10,0.01)
sigma = 1/(1 + np.exp(-t))
plt.plot(t,sigma)

import numpy as np
import matplotlib.pyplot as plt
import torch

num_pts = 500 #Total number of points
d = 2 #We are in dimension 2
m = int(num_pts/2) #Number in each class
data = np.random.randn(m,2) - [2,2]
data = np.vstack((data,np.random.randn(m,2) + [4,4]))
target = np.hstack((np.zeros((m,)),np.ones((m,))))

#Scatter plot the points colored by class
plt.scatter(data[:,0],data[:,1],c=target, cmap=plt.cm.Paired,edgecolors='black')
plt.savefig('clusters.eps')

#Convert to torch
data = torch.from_numpy(data).float()
target = torch.from_numpy(target)

import torch
import torch.optim as optim
import torch.nn.functional as F

# Create random Tensors for weight and bias
w = torch.randn(d, requires_grad=True)
b = torch.randn(1, requires_grad=True)

#Use an optimizer so we can avoid explicitly coding gradient descent.
#Need to provide a list of the parameters to be optimized over and the learning rate.
optimizer = optim.Adam([w,b], lr=1)  #Learning rate

for i in range(500):
    #Set the gradients to zero (in place of a.grad = None, etc.)
    optimizer.zero_grad()

    # Forward pass: compute predicted y using operations on Tensors (data@w is matrix/vector multiplication)
    output = torch.sigmoid(data@w + b)

    # Compute the loss using operations on Tensors.
    #loss = (output - target).pow(2).sum()
    loss = torch.sum((output - target)**2)

    #Print iteration and loss
    print('Iter:%d, Loss:%.4f'%(i,loss.item()))

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    loss.backward()

    #Take a step of gradient descent
    optimizer.step()

#Plotting a decision boundary
X,Y = np.mgrid[-7:7:0.01,-7:7:0.01]
points = np.c_[X.ravel(),Y.ravel()]

#Detach from autograd and convert back to numpy
w_npy = w.detach().numpy()
b_npy = b.detach().numpy()
data_npy = data.numpy()

#Predict class using model
y = data_npy@w_npy + b_npy > 0
z = points@w_npy + b_npy > 0

plt.figure()
plt.scatter(data_npy[:,0],data_npy[:,1],zorder=2,c=y,cmap=plt.cm.Paired,edgecolors='black')
C = plt.contourf(X, Y, z.reshape(X.shape), cmap=plt.cm.Paired,zorder=1)
plt.savefig('clusters_classified.eps')
print(w_npy)
print(b_npy)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

#Our neural network as 2 layers with 100 hidden nodes
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2,1000) #include weights & bias
        self.fc2 = nn.Linear(1000,2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

#Training data on circles
x,y = datasets.make_circles(n_samples=200,noise=0.1,factor=0.3)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.Paired,edgecolors='black')
plt.savefig('tworings.eps')

#Convert to torch
data = torch.from_numpy(x).float()
target = torch.from_numpy(y)

#Setup model
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)  #Learning rate

#Precompute for plotting
X,Y = np.mgrid[-1.5:1.5:0.01,-1.5:1.5:0.01]
points = torch.from_numpy(np.c_[X.ravel(),Y.ravel()]).float()

#Training
for i in range(200):
    if i % 10 == 0:
        print('Iter:%d'%i)
        model.eval()
        #Plot the classification decision boundary
        with torch.no_grad(): #Tell torch to stop keeping track of gradients
            plt.figure()
            plt.scatter(x[:,0],x[:,1],zorder=2,c=y,cmap=plt.cm.Paired,edgecolors='black')
            plt.axis('off')
            Z = np.argmax(model(points).numpy(),axis=1).reshape(X.shape)
            C = plt.contourf(X, Y, Z, cmap=plt.cm.Paired,zorder=1)

    #Training mode, run data through neural network
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model(data), target)

    #Back propagation to compute all gradients, and a single gradient descent step
    loss.backward()
    optimizer.step()

#Save our neural network model so we can load again later
torch.save(model.state_dict(), "ring_classify.pt")

#Application
pip install -q graphlearning

import graphlearning as gl

#Load MNIST data
x,y = gl.datasets.load('mnist',metric='raw')

#Display images
gl.utils.image_grid(x,n_rows=16,n_cols=16)
print(x.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#Our neural network as 2 layers with 32 hidden nodes
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,32)
        self.fc2 = nn.Linear(32,10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

 #Training and testing data, converted to torch
train_size = 60000
data = torch.from_numpy(x[:train_size,:]).float()
target = torch.from_numpy(y[:train_size]).long()
data_test = torch.from_numpy(x[train_size:,:]).float()
target_test = torch.from_numpy(y[train_size:]).long()

#Setup model
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.1)  #Learning rate

#Training
for i in range(1000):

    #Training mode, run data through neural network
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model(data), target)

    #Back propagation to compute all gradients, and a single gradient descent step
    loss.backward()
    optimizer.step()

    #Accuracy
    model.eval()
    with torch.no_grad():
        test_pred = torch.argmax(model(data_test),axis=1)
        test_accuracy = torch.mean((test_pred == target_test).float())
        train_pred = torch.argmax(model(data),axis=1)
        train_accuracy = torch.mean((train_pred == target).float())
        print('Iter:%d, Test Accuracy=%.2f, Training Accuracy=%.2f'%(i,test_accuracy*100,train_accuracy*100))

#Save our neural network model so we can load again later
torch.save(model.state_dict(), "mnist_classify.pt")

#Exercises
#This sets up Kaggle to download directly to Google Colab (only needs to run once)
!mkdir -p ~/.kaggle
!wget https://www-users.math.umn.edu/~jwcalder/MCFAM/kaggle.json -P ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

#As an example, we now download and unzip the Sign Language MNIST dataset from Kaggle
import zipfile

!kaggle datasets download -d datamunge/sign-language-mnist
with zipfile.ZipFile("sign-language-mnist.zip","r") as zip_ref:
    zip_ref.extractall(".")
!ls -l

#Let's load the training data with pandas
import pandas as pd

df_train = pd.read_csv('sign_mnist_train.csv')
df_train

import graphlearning as gl
import torch
import numpy as np

#The first column is the label and the rest are the pixel values
df_train = pd.read_csv('sign_mnist_train.csv')
train_labels = df_train.values[:,0]
train_data = df_train.values[:,1:]

df_train = pd.read_csv('sign_mnist_test.csv')
test_labels = df_train.values[:,0]
test_data = df_train.values[:,1:]

print(train_data.shape)
print(test_data.shape)

#Display image grid with graphlearning
gl.utils.image_grid(train_data)

#Sign Language MNIST can also be loaded from the GraphLearning package
data,labels = gl.datasets.load('signmnist')

print(labels)
gl.utils.image_grid(train_data)
