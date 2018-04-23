import sys
# import LIBSVM
# sys.path.insert(0, "/Users/goodhat/Downloads/libsvm-3.22/python")

import numpy as np
from cvxopt import matrix, solvers
from svmutil import *

N = 7 #data points
D = 2 #dimension

x = [[1,0], [0,1], [0,-1], [-1,0], [0,2], [0,-2], [-2,0]]   # original data points
y = [0, 0, 0, 1, 1, 1, 1]  # labels

# x = np.array(x)
# y = np.array(y)



def SVM(x):
    # deal with transformation
    z = np.zeros((N,D))
    for i in range(7):
        z[i][0] = 2 * x[i][1]**2 - 4 * x[i][0] + 2
        z[i][1] = x[i][0]**2 - 2 * x[i][1] - 1
    print(z)

    Q = np.concatenate((np.zeros((1,D)), np.identity(D)), axis=0)
    Q = np.concatenate((np.zeros((D+1,1)), Q), axis=1)
    p = np.array([[0.], [0.], [0.]])

    G = np.reshape(y[0]*np.append([1], z[0]), (1,3))
    for i in range(N-1):
        G = np.concatenate((G, np.reshape(y[i+1]*np.append([1], z[i+1]), (1,3))), axis=0)

    G = (-1)*G.astype('double')
    h = np.ones((N,1))*(-1)


    Q = matrix(Q)
    p = matrix(p)
    G = matrix(G)
    h = matrix(h)
    sol=solvers.qp(Q, p, G, h)
    print(sol['x'])

    return sol


def kernel(x1, x2):
    return (1 + 2 * x1.dot(x2))**2


def dual_kernel_SVM(x):
    model = svm_train(y, x, '-h 0 -t 1 -g 2 -r 1 -d 2')
    svm_save_model("1", model)
    # calculate the six arguments for QP solver
    '''
    p = np.ones((N,1))*(-1)
    Q = np.empty([N,N])
    for i in range(N):
        for j in range(N):
            Q[i,j] = y[i] * y[j] * kernel(x[i], x[j])
            # print(i, j, kernel(x[i], x[j]))
    print(Q)
    G = np.identity(N) * (-1)
    h = np.zeros((N,1))
    A = y.reshape((1,N))
    A = A.astype('double')
    b = np.array([[0.]])

    # numpy to matrix
    Q = matrix(Q)
    p = matrix(p)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    sol=solvers.qp(Q, p, G, h, A, b)
    alpha = sol['x'] # lagrange multipliers

    print (sol['x'])

    w = np.zeros((1,2))

    for i in range(N):
        w =  w + sol['x'][i] * y[i] * z[:,i]

    bias = y[1] + sum(alpha[1] * y [1] * kernel(x[:,], x[1]))
    print (bias)
    '''
    return



if __name__ == '__main__':
    dual_kernel_SVM(x)
