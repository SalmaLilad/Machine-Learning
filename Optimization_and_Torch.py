#gradient descent, explicitly
#Function you wish to minimize
def f(x,y):
    return (1/2)*x**2 + (1/4)*y**4

#Gradient of f (vector of derivatives in x and y)
def gradient_f(x,y):
    return x, y**3

x,y = 1,2 #Initial values of x=1 and y=2

dt = 0.1 #alpha
for i in range(100):
    grad = gradient_f(x,y)
    print("(x,y)=(%.3f,%.3f), f(x,y)=%.3f, Grad f(x,y)=(%.5f,%.5f)"%(x,y,f(x,y),grad[0],grad[1]))
    # x -= dt*grad[0]   # x -= a is the same as x = x - a
    # y -= dt*grad[1]

    x = x - dt * grad[0]
    y = y - dt * grad[1]

import matplotlib.pyplot as plt

x,y = 2,0  #Initial values of x=1 and y=2

plt.text(x,y,f'Initial point=({x},{y})',horizontalalignment='right')
dt = 0.01
N  = 1000
for i in range(N):
    x_old,y_old = x,y
    grad = gradient_f(x,y)
    x -= dt*grad[0]
    y -= dt*grad[1]
    plt.plot([x_old,x],[y_old,y],'cyan',alpha=0.9*(i+1)/N + 0.1)
    if i % 20 == 0:
      plt.scatter(x,y,c='blue',marker='.')

plt.text(x,y-0.1,f'Final point=({x:.2f},{y:.2f})',horizontalalignment='right')

plt.xlim((-1.2,1.2))
plt.ylim((-1.2,2.2))
plt.scatter(0,0)
plt.text(0,0,f'Minimizer=(0,0)',color='red',horizontalalignment='right')

#optimization in torch
import torch

x = torch.tensor([2.0], requires_grad = True)
z = x**3
z.backward() #Computes the gradient
print(x.grad.data) #Prints gradient
print(3*x**2) #Compare against true gradient

import torch

#Need to define x and y as tensors with requires_grad=True
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

#The time step dt is called a learning rate in machine learning
learning_rate = 0.1
for t in range(100):

    # Forward pass: Compute the function you wish to minimize
    loss = f(x,y)

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call x.grad and y.grad will be Tensors holding
    # the gradient of the loss with respect to x and y respectively.
    loss.backward()

    # Manually update (x,y) using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        x -= learning_rate * x.grad
        y -= learning_rate * y.grad

        #Print info
        print("(x,y)=(%.3f,%.3f), f(x,y)=%.3f, Grad f(x,y)=(%.5f,%.5f)"%(x,y,f(x,y),x.grad.data,y.grad))

        # Manually zero the gradients after updating weights
        x.grad = None
        y.grad = None

#application: polynomial fitting
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-np.pi, np.pi, 2000)

# Create random Tensors for weights. For a third order polynomial, we need
# 5 weights: y = a + b x + c x^2 + d x^3 + e x^4
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
c = torch.randn(1, requires_grad=True)
d = torch.randn(1, requires_grad=True)
e = torch.randn(1, requires_grad=True)

#Use an optimizer so we can avoid explicitly coding gradient descent.
#Need to provide a list of the parameters to be optimized over and the learning rate.
optimizer = optim.Adam([a,b,c,d,e], lr=1e-1)  #Learning rate

for t in range(10000):
    #Set the gradients to zero (in place of a.grad = None, etc.)
    optimizer.zero_grad()

    # Forward pass: compute predicted y using operations on Tensors.
    y = a + b * x + c * x ** 2 + d * x ** 3 + e * x**4

    # Compute the loss using operations on Tensors.
    loss = torch.sum(torch.abs(y - torch.sin(x))**2)

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    loss.backward()

    #Take a step of gradient descent
    optimizer.step()

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3 + {e.item()} x^4')
plt.plot(x,torch.sin(x),label='Sin(x)')
plt.plot(x,y.detach(),label='Polynomial approximation') # Try this without .detach() to see why we need it.
plt.legend()

!pip install torch

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-np.pi, np.pi, 2000)

# Create random Tensors for weights. For a third order polynomial, we need
# 5 weights: y = a + b x + c x^2 + d x^3 + e x^4
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.

n_sin = 100

a_sin = torch.randn(n_sin, requires_grad=True)




#Use an optimizer so we can avoid explicitly coding gradient descent.
#Need to provide a list of the parameters to be optimized over and the learning rate.
optimizer = optim.Adam([a_sin], lr=1e-1)  #Learning rate

for t in range(5000):
    #Set the gradients to zero (in place of a.grad = None, etc.)
    optimizer.zero_grad()

    # Forward pass: compute predicted y using operations on Tensors.
    y = 0
    for i in range(n_sin):
      y = y + a_sin[i]*torch.sin((i+1)*x)


    # Fill in your own code here, using the following steps as a guide:
      # Compute the loss using operations on Tensors.
    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    #Take a step of gradient descent
$print(f'Result: y = {a.item()}sin(x) + {b.item()} sin(2x) + {c.item()} sin(3x) + {d.item()} sin(4x) + {e.item()} sin(5x)')
plt.plot(x,x,label='y=x')
plt.plot(x,y.detach(),label='Trig Polynomial Approximation') # Try this without .detach() to see why we need it.
plt.legend()

#Exercise: higher order trig polynomials
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch

class net(nn.Module):
    def __init__(self, n):
        super(net, self).__init__()
        self.n = n
        self.p1 = nn.Parameter(torch.randn(1,n))
        self.I = torch.arange(n)[:,None].float()

    def forward(self, x):
        y = self.p1@torch.sin(self.I@x)
        return y

model = net(500)

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-np.pi, np.pi, 2000)
x = x[None,:]

# Create random Tensors for weights. For a third order polynomial, we need
# 5 weights: y = a + b x + c x^2 + d x^3 + e x^4
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.


#Use an optimizer so we can avoid explicitly coding gradient descent.
#Need to provide a list of the parameters to be optimized over and the learning rate.
optimizer = optim.Adam(model.parameters(), lr=1e-1)  #Learning rate

for t in range(5000):
    # Train the model; fill in your own code here
