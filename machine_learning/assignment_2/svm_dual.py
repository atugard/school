import numpy as np
from cvxopt import solvers,matrix
from sklearn import datasets
from math import sqrt

#Simple function to calculate norm of a vector.
def norm(w):
    return sqrt(np.dot(w,w))

iris = datasets.load_iris()
xs = iris['data']
ys = iris['target']

#label iris-setosa -1, and others +1
for i in range(len(ys)):
    if(ys[i] == 0):
        #Iris-setosa is -1
        ys[i] = -1
    else:
        #The rest are 1
        ys[i]=1


#returns the subdataset of xs of properties i,j
def pairs(i,j,xs):
    res = np.empty((0,2))
    for x in xs:
        res = np.vstack((res,np.array([x[i],x[j]])))
    return res

        
#Here is a link to see the QP set up https://cvxopt.org/userguide/coneprog.html#quadratic-programming

#We minimize 1/2 sum( alpha_i *  alpha_j * (y_i x_i) * (y_j x_j)) - sum ( a_i ),
#such that 0 <= alpha_i <= C
#          sum ( y_i * alpha_i ) = 0

# P= (y_i x_i * y_j x_j),
def P(xs, ys):
    _P = np.empty((len(xs[0]),0))
    for (x,y) in zip(xs,ys):
        _P = np.hstack((_P, (x * y).reshape(-1,1))) # there's gotta be a prettier way to do this, but it works...
    return matrix(np.dot(_P.T,_P))

# q = (-1, -1, ..., -1)^T
def q(m):
    return matrix(-1*np.ones(m))

#This is just two mxm identity matrices on top of each other, with the first scaled by -1.
# G = ((-1, 0, ..., 0)
#      (0, -1, ..., 0)
#      (0, 0, ..., -1)
#      (1, 0, ..., 0)
#      (0, 1, ..., 0)
#      (0,0, ..., 1))
def G(m):
    _G = -1*np.identity(m)
    _G = np.vstack((_G, np.identity(m)))
    return matrix(_G)
              


#m zeroes, then m C's.
#h = (0, 0, ..., 0,C, ..., C)^T
def h(m, C):
    _h = np.ones(2*m)
    for i in range (0, m):
        _h[i] = 0
    for i in range (m, 2*m):
        _h[i] = C
    return matrix(_h)

#A = ys
def A(ys):
    return matrix(ys, (1,len(ys)), 'd')

#b = (0)
b = matrix(0,(1,1), 'd')

def solve(xs,ys,C):
    m = len(xs)

    alpha = solvers.qp(P(xs,ys), q(m), G(m), h(m, C), A(ys), b)['x']
    
    w = np.zeros(len(xs[0]))
    
    sv_indices = []
    support_vectors = []
    svs = []
    soln = {}

    #set small alpha's to zero, and save non-zero alpha index to sv_indices
    for i in range(len(alpha)):
        if(abs(alpha[i]) < 0.01):
            alpha[i] = 0
        else:
            sv_indices.append(i)
            
    #calculate w
    for i in sv_indices:
        w = w + (alpha[i] * ys[i] * np.array(xs[i]))
        support_vectors.append(xs[i])
        
    # to calculate b = y_i - sum_j (alpha_j y_j (x_i * x_j))
    # just take i = sv_indices[0], then
    i = sv_indices[0]
    b_svm = ys[i]
    for j in sv_indices:
        b_svm = b_svm - (alpha[j] * ys[j] * np.dot(xs[j], xs[i]))

    # For any support vector x with correspond 0 < alpha < C we can get the margin by
    # abs ( w * x + b) / norm(w), so
    margin = 0
    for a in alpha:
        if (a > 0 and a < C):
            margin = (abs( np.dot(w,xs[i]) + b_svm))/norm(w)
    soln.update({'w' : w,
                 'b' : b_svm,
                 'margin' : margin,
                 'Support vectors': support_vectors
                 })
    return soln



#Data from Question 3 part C of assignment:
testxs = np.array([[1,1], [2,2],[0,2], [0,1],[1,0],[-1,0]])
testys = np.array([1,1,1,-1,-1,-1])

#Uncomment and run python -i svm_dual.py
soln = solve(testxs,testys,5)
#soln = solve(pairs(0,1, xs),ys,5)
#soln = solve(pairs(0,2, xs),ys,5)
#soln = solve(pairs(0,3, xs),ys,5)
#soln = solve(pairs(1,2, xs),ys,5)
#soln = solve(pairs(1,3, xs),ys,5)
#soln = solve(pairs(2,3, xs),ys,5)


