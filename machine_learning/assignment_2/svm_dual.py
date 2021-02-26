import numpy as np
from qpsolvers import solve_qp
import csv
from cvxopt import solvers

# read in data
# raw input is organized like ['TYPE', 'PW', 'PL', 'SW', 'SL']
xs = []
ys = []
with open('iris.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    next(csvReader, None)
    for row in csvReader:
        xs.append(list(map(int,row[1:])))
        ys.append(int(row[0]))

#Let m be the sample size, and d be the input dimension. 

#We look for min_alpha 1/2 sum[alpha_i alpha_j (y_i x_i)*(y_j x_j) | i <- [1..m], j <- [1..m]] - sum[alpha_i | i <- [1..n]]
#such that 1) 0 <= alpha <= C
#          2) alpha_1 y_1 + ... + alpha_n y_n = 0

#The standard quadratic programming objective function is min_alpha (1/2)(alpha^T)P(alpha) + (q^T)alpha
#such that G(alpha) <= h     (we won't need this)
#          A(alpha) = b      (corresponds to 2 above)
#          lb <= alpha <= ub (corresponds to 1 above)

#P = (y_i x_i * y_j x_j)_{ij} is the Gram Matrix. We require that P be positive definite, which I don't think P will be for our data.
#We can add epsilon * Identity, for epsilon some small constant, to make it positive definite.
def P(inputs, labels,epsilon):
    _gram = np.empty((len(inputs[0]),0))
    for (label,inp) in zip(labels,inputs):
        _gram = np.hstack((_gram, (label * np.array(inp)).reshape(-1,1)))
    _gram = np.dot(_gram.T , _gram)
    return _gram + epsilon * np.identity(len(_gram))
        
        
        
#q = (-1,-1,...,-1), m dimensional.
def q(m):
    return (-1)*np.ones(m)

#A = ys
#b = 0

#We have lb <= alpha <= ub,
#where lb=(0,...,0), m-dimensional
#      ub=(C,...,C), m-dimensional
def lb(m):
    return np.zeros(m)

def ub(m,C):
    return C * np.ones(m)

soln = solve_qp(P(xs,ys, 0.00001), q(len(xs)), A=np.array(ys), b=0, lb=lb(len(xs)), ub=ub(len(xs), 2))

