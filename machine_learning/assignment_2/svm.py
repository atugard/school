import numpy as np
from cvxopt import solvers,matrix
from sklearn import datasets
from math import sqrt


#===============HELPER FUNCTIONS===============
#Simple function to calculate norm of a vector.
def norm(w):
    return sqrt(np.dot(w,w))

#returns pairs sub dataset of xs of pairs i,j.
def pairs(i,j,xs):
    res = np.empty((0,2))
    for x in xs:
        res = np.vstack((res,np.array([x[i],x[j]])))
    return res

#===============QUADRATIC PROGRAMMING===============
        
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

def solve(data,C):
    xs = data[0]
    ys = data[1]
    
    m = len(xs)
    alpha = solvers.qp(P(xs,ys), q(m), G(m), h(m, C), A(ys), b)['x']
    
    sv_indices = []

    #data to be computed and returned
    w = np.zeros(len(xs[0]))
    support_vectors = []
    b_svm  = 0
    margin = 0
    
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
        
    #calculate b, margin
    for i in sv_indices:
        if(alpha[i] < C):
            b_svm = ys[i]
            for j in sv_indices:
                b_svm = b_svm - (alpha[j] * ys[j] * np.dot(xs[j], xs[i]))
            margin = abs(np.dot(w,xs[i]) + b_svm)/norm(w)
            break

    return {'w' : w,
            'b' : b_svm,
            'margin' : margin,
            'Minimal support vectors': support_vectors
            }


#===============DATA===============

#Iris data

iris = datasets.load_iris()
iris_data= [iris['data'],
            iris['target']]

#label iris-setosa -1, and others +1.
for i in range(len(iris_data[1])):
    if(iris_data[1][i] == 0):
        #Iris-setosa is -1
        iris_data[1][i] = -1
    else:
        #The rest are 1
        iris_data[1][i]=1

#Data from Question 2 of assignment:
q2_data = [np.array([[-1],[-0.8],[1]]),
           np.array([-1,1,1])]

#Data from Question 3 part C of assignment:
q3c_data = [np.array([[1,1], [2,2],[0,2], [0,1],[1,0],[-1,0]]),
            np.array([1,1,1,-1,-1,-1])]



#===============SVM COMPUTATION===============

#Constant value for expressions below
C = 10000000

#Uncomment and run python -i svm.py
#solnq2  = solve(q2_data,C)
#solnq3c = solve(q3c_data,C)
#soln    = solve(iris_data,C)
#soln01  = solve([pairs(0,1, iris_data[0]),iris_data[1]],C)
#soln02  = solve([pairs(0,2, iris_data[0]),iris_data[1]],C)
#soln03  = solve([pairs(0,3, iris_data[0]),iris_data[1]],C)
#soln12  = solve([pairs(1,2, iris_data[0]),iris_data[1]],C)
#soln13  = solve([pairs(1,3, iris_data[0]),iris_data[1]],C)
#soln23  = solve([pairs(2,3, iris_data[0]),iris_data[1]],C)





