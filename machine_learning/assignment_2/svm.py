import numpy as np
from qpsolvers import solve_qp
import csv

# read in data
# raw input is organized like ['TYPE', 'PW', 'PL', 'SW', 'SL']
xs = []
ys = []
with open('iris.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    next(csvReader, None)
    for row in csvReader:
        xs.append(row[1:])
        ys.append(row[0])
        
#svm solver of the form minimize 1/2(X^T)PX + q^TX subject to Gx <= h, Ax = b, lb <= x <= ub,
#In our case we minimize 1/2 (w^T)w + C\sum_{i=1}^m\xi_i over (w,b,xi), subject to ...
#Thus, in our case our P=diag(1,1,...,1,0,...,0), which has 1's on the diagonal up to the dimension of w, and zeroes afterwards.
#Thus if we let x=(w,b,xi), we get 1/2(x^T)Px = 1/2(magnitude(w))
#and q=(0,0,...,0,C,C,...,C), where we have m+1 leading zeroes, then all C's. Then q^Tx = \sum(C*(xi_i))

#Okay, well, let's get started.

#P in the optimization problem, d is such that w lives in R^d. n is the number of data points in the sample.
#(x^T)Px = (1/2)magnitude(w)^2
#This matrix is not strictly positive semi-definite, added epsilon term to make it so.
def P(d, n, epsilon):
    _P = np.identity(d+n+1)
    for i in range(d, d+n+1):
        _P[i][i] = epsilon
    return _P

#Q in the optimization problem, d is the dimension that w lives in, n is the number of data points in the sample (and therefore the number of xi_i's)
#q^Tx = C\sum(xi_i)
def q(d, n , C):
    _q = np.ones(d+n+1) * C
    for i in range(0, d+1):
        _q[i] = 0
    return _q    

#Gx = -xi_i -y_i(w*x_i + b), so we need the data to implement G, where x=(w_1, \dots, w_d, b, xi_1, dots, xi_m)
def G(_xs, _ys):

    #Dimension of w 
    d=len(_xs[0])

    #Number of sample points, also number of constraints xi_j
    n=len(_xs)
    
    _G = np.empty((0, d+1))
    for datum in zip(_xs,_ys):
         label = datum[1]
         x = datum[0]
         x=np.array(list(map(int,x))) #convert the chars in x to integers
         row = int(label) * np.append(x,1)
         _G = np.vstack((_G, row))
    return np.block([-1*_G,-1*np.identity(len(_G))])

#Gx <= h if and only if y_i(w*x_i + b) >= 1 -xi_i
def h(n):
    return -1*np.ones(n)

#lower bound on x. I only need to constrain the xi's to be non-negative, but this algorithm requires constraints on all values
#thus I'll just put the lowest value possible for w_1, ..., w_d, b, and then 0 for the xi's
def lb(d,n):
    _lb = np.ones(d+n+1)
    for i in range(0,d+1):
        #I think this is the minimal integer value. This makes it so (w,b) can be anything, as they are the first d+1 entries (d for w, and then 1 for b)
        _lb[i] = -9223372036854775807
    for i in range(d+1,d+n+1):
        #This constrains the xi's to be non-negative.
        _lb[i] = 0
    return _lb

def solve(C, xs, ys):
    d = len(xs[0])
    n = len(xs)
    return solve_qp(P(d,n, 10**(-100)), q(d,n, C), G(xs,ys), h(n),lb=lb(d,n)) 
    

