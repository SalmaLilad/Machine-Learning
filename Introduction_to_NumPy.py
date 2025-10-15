import numpy
import numpy as np
from numpy import array
from numpy import sin

#arrays
a = np.array([1,2,3])
b = np.ones(10)
c = np.zeros((3,))
print(a)
print(b)
print(c)


d = np.array([[1,2,3,4,5],[6,7,8,9,10]])  #2D array of size 2x5, initialized manually
e = np.ones((2,5))          #2D ones array of size 2x5
f = np.zeros((2,2))       #3D zeros array of size 2x5x8
np.random.seed(10)
g = np.random.rand(2,6)     #Random 2D array of size 2x8
print(d)
print(e)
print(f)
print(g)
print(d.shape)

x = np.random.rand(3,5)
print(x)
print(x[0,0])
print(x[0,1])
print(x[1,0])

x[0,0] = np.pi
print(x)

#slicing arrays
A = np.random.rand(5,3)
print(A)
print(A[:,0]) #First column of A
print(A[0,:]) #First row of A

a = np.arange(0,19,1) #Array of numbers from 0 to 18=19-1 going by 1
print(a)
print(a[0:7])
print(a[7:19])
print(a[10:-2])  #Note the -2 means 2 before the end of the array
print(a[::3])

a = np.linspace(0,19,100) #Array of 100 numbers from 0 to 19 equally spaced
print(a)

A = np.reshape(np.arange(30),(3,10))
print(A)
print(A[:,::2])
print(A[0:2,:])
print(A[0,:])
print(A[:,[3]])

#logical indexing
A = np.random.rand(3,5)
print(A)

I = A > 0.5  #Boolean true/false
print(I)
#A[A>0.5] = 10
A[I] = 10
print(A)
A[~I] = 0  #The ~ negates a boolean array
print(A)

#operations on numpy arrays
import numpy as np
import time

#Let's make two long lists that we wish to add elementwise
n = 300000
A = n*[1]
B = n*[2]
print('A=',end='');print(A)
print('B=',end='');print(B)

#Let's add A and B elementwise using a loop in Python
start_time = time.time()
C = n*[0]
for i in range(n):
    C[i] = A[i] + B[i]
python_time_taken = time.time() - start_time
print('C=',end='');print(C)
print("Python took %s seconds." % python_time_taken)

#Let's convert to NumPy and add using NumPy operations
A = np.array(A)
B = np.array(B)

start_time = time.time()
C = A + B
numpy_time_taken = time.time() - start_time
print("NumPy took %s seconds." % (numpy_time_taken))

print('NumPy was %f times faster.'%(python_time_taken/numpy_time_taken))

import numpy as np

A = np.random.rand(3,5)
B = np.random.rand(3,5)

print('A*B=',end='');print(A*B)  #elementwise multiplication
print('A-B=',end='');print(A-B)  #elementwise subtraction

#Examples of matrix multiplication and matrix/vector multiplication
print('A@B.T=',end='');print(A@B.T)   #B.T means the transpose of B
C = np.random.rand(5,7)
D = np.ones((5,))
print('A@C=',end='');print(A@C)
print('A@D=',end='');print(A@D)

import numpy as np
import scipy.sparse as sparse

A = np.random.rand(3,3)
x = np.random.rand(3,1)

print(A@x)
print(A*x)

A = np.matrix(A)
print(A*x)

A = sparse.csr_matrix(A)
print(A*x)

import numpy as np

A = np.reshape(np.arange(10),(2,5))

print(A)
print(A**2) #Square all elements in A
print(np.sin(A)) #Apply sin to all elements of A
print(np.sqrt(A)) #Square root of all elements of A

import numpy as np

A = np.reshape(np.arange(30),(3,10))

print(A)
print(np.sum(A))   #Sums all entries in A
print(np.max(A,axis=0))  #Gives sums along axis=0, so it reports column sums
print(np.sum(A,axis=1))  #Row sums

#broadcasting
import numpy as np

A = np.reshape(np.arange(30),(3,10))
print(A)

x = np.arange(10)
print(x.shape)
print(x)
print(A+x) #Adds the row vector of all ones to each row of A
print(A+1)

import numpy as np

A = np.reshape(np.arange(30),(3,10))
print(A)

x = np.ones((3,))
print(x)
print(A+x[:,None]) #Adds the row column of all ones to each row of A

print(x)
print(x[:,None])
print(x.shape)
print(x[:,None].shape)
print(A+x[:,None])
print(A+A[:,0][:,None])

#exercises
def inner_product(x,y,C):
  return x.T @ C @ y

testx = np.random.rand(4)
testy = np.arange(4)
testC = np.ones((4,4))
inner_product(testx, testy, testC)

def approx_pi(N):
    n = np.arange(1,N+1)
    return np.sqrt(6*np.sum(1/n**2))
print(approx_pi(100000000))

import numpy as np
n = 10**7
dx = 1/n
x = np.arange(0,1,dx)

print(x.shape, x)

#left point rule
lpr = 4*np.sum(np.sqrt(1 - x**2)*dx)
print('Left Point Rule',lpr,'Error',abs(lpr-np.pi))

#right point rule
rpr = 4*np.sum(np.sqrt(1 - (x+dx)**2)*dx)
print('Right Point Rule',rpr,'Error',abs(rpr-np.pi))

#mid point rule
mpr = 4*np.sum(np.sqrt(1 - (x+dx/2)**2)*dx)
print('Mid Point Rule',mpr,'Error',abs(mpr-np.pi))

#trapezoid point rule
tr = 4*np.sum((np.sqrt(1 - (x+dx)**2) + np.sqrt(1 - x**2))*dx/2)
print('Trapezoid Rule',tr,'Error',abs(tr-np.pi))

#Simpsons's rule
sr = (2*mpr + tr)/3
print('Simpsons Rule',sr,'Error',abs(sr-np.pi))

e,v = np.linalg.eig([[1,1],[1,0]])
print(v,e)

def power_method(A,T):
    x = np.random.rand(A.shape[0],1)
    for i in range(T):
        y = A@x
        x = y/np.linalg.norm(y)
    return x, x.T@A@x

#Let's do the Fibonacci example
A = np.array([[1,1],[1,0]])
x,l = power_method(A,1000)
print(x,l)

x = np.arange(5)[:,None]
print(x.T)
print(x)
print(x.T + x)

def prime_sieve(n):

    primes = [] #Empty list for primes
    prime_mask = np.ones(n+1,dtype=bool)
    prime_mask[0] = False
    prime_mask[1] = False

    c = np.argmax(prime_mask)
    while c > 0:
        primes += [c]
        prime_mask[c::c] = False
        c = np.argmax(prime_mask)
    return primes

#Find all prime numbers up to 10^6.
primes = prime_sieve(10**6)
for p in primes:
    print(p)
