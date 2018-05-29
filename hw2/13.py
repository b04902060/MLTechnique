import numpy as np
import csv
import math
from numpy.linalg import inv


gamma = [32, 2, 0.125]
lamb = [0.01, 0.1, 1, 10, 100]

def loadData(filepath):
    X = []
    Y = []
    file = csv.reader(open(filepath, "r"), delimiter = ' ')
    for r in file:
        X.append([1.])
        X[-1].extend([float(r[i+1]) for i in range(10)])
        Y.append(int(r[-1]))
    return np.array(X), np.array(Y)

def rbf(gamma, x1,x2):
    return np.exp(-gamma*((x1-x2).dot(x1-x2)))

def get_kernel_matrix(X, gamma):
    N = X.shape[0]
    K = np.empty([N, N])

    for i in range(N):
        for j in range(N):
            K[i][j] = rbf(gamma, X[i], X[j])
    return K

def evaluate(W, X_test, Y_test):
    ans = 0
    for test, test_ans in zip(X_test,Y_test):
        sum = W.dot(test)
        predict = -1 if sum<0 else 1
        ans += 1 if predict != test_ans else 0
    return ans


def solve(lamb, X_train, Y_train, X_test, Y_test):
    # W = (X^t*X + lambda*I)^(-1) * X^t * y
    W = inv((X_train.transpose().dot(X_train))+lamb*np.eye(X_train.shape[1])).dot(X_train.transpose().dot(Y_train))

    print ("========  lambda:", lamb, "\t========")
    print ("Ein: ", (evaluate(W, X_train, Y_train)/400.))
    print ("Eout:", (evaluate(W, X_test, Y_test)/100.))


def main():
    X,Y = loadData("hw2_lssvm_all.dat.txt")
    X_train = X[:400]
    Y_train = Y[:400]
    X_test  = X[400:]
    Y_test  = Y[400:]

    for l in lamb:
        solve(l, X_train, Y_train, X_test, Y_test)



if __name__ == '__main__':
    main()
