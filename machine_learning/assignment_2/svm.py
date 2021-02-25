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
def P(d, n):
    _P = np.identity(d+n+1)
    for i in range(d, d+n+1):
        _P[i][i] = 0
    return _P

#Q in the optimization problem, d is the dimension that w lives in, n is the number of data points in the sample (and therefore the number of xi_i's)
#q^Tx = C\sum(xi_i)
def q(d, n , C):
    _q = np.identity(d+n+1) * C
    for i in range(0, d+1):
        _q[i][i] = 0
    return _q    

#Gx = -xi_i -y_i(w*x_i + b), so we need the data to implement G, where x=(w_1, \dots, w_d, b, xi_1, dots, xi_m)
def G(data):
    print("tbd")

#takes (x_i, y_i)    
#def handleRows(datum)

    

