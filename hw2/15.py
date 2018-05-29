import numpy as np
import csv
import math
from numpy.linalg import inv
import random


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

def evaluate(W_matrix, X_test, Y_test):
    Y_vote = 0
    ans = 0
    for test, test_ans in zip(X_test,Y_test):
        Y_vote = 0
        for w in W_matrix:
            sum = w.transpose().dot(test.reshape(11,1))
            predict = -1 if sum<0 else 1
            Y_vote += predict
        ans += 1 if Y_vote*test_ans<0 else 0

    return ans


def bootstrap(X,Y):
    rand = [random.randint(0,399) for i in range(400)]
    #print(rand)
    X_samples = [X[r] for r in rand]
    Y_samples = [Y[r] for r in rand]
    return np.array(X_samples), np.array(Y_samples).reshape((400,1))

def solve(lamb, X_train, Y_train, X_test, Y_test):
    W_matrix = []
    # W = (X^t*X + lambda*I)^(-1) * X^t * y
    for i in range(2500):
        X_samples, Y_samples = bootstrap(X_train, Y_train)
        W = inv((X_samples.transpose().dot(X_samples))+lamb*np.eye(X_samples.shape[1])).dot(X_samples.transpose().dot(Y_samples))
        W_matrix.append(W)
    

    print ("========  lambda:", lamb, "\t========")
    print ("Ein: ", (evaluate(W_matrix, X_train, Y_train)/400.))
    print ("Eout:", (evaluate(W_matrix, X_test, Y_test)/100.))


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
