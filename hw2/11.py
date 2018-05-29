import numpy as np
import csv
import math
from numpy.linalg import inv


gamma = [32, 2, 0.125]
lamb = [0.001, 1, 1000]

def loadData(filepath):
    X = []
    Y = []
    file = csv.reader(open(filepath, "r"), delimiter = ' ')
    for r in file:
        X.append([float(r[i+1]) for i in range(10)])
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

def evaluate(gamma, beta, X_train, X_test, Y_test):
    ans = 0
    for test, test_ans in zip(X_test,Y_test):
        sum = 0.
        for b, train in zip(beta,X_train):
            sum += b*rbf(gamma,test,train)
        predict = -1 if sum<0 else 1
        ans += 1 if predict != test_ans else 0
    return ans


def solve(gamma, lamb, X_train, Y_train, X_test, Y_test):
    K = get_kernel_matrix(X_train, gamma)
    beta = inv(lamb*np.eye(400)+get_kernel_matrix(X_train, gamma)).dot(Y_train)

    print ("========\tgamma:", gamma, " lambda:", lamb, "\t========")
    print ("Ein: ", (evaluate(gamma, beta, X_train, X_train, Y_train)/400.))
    print ("Eout:", (evaluate(gamma, beta, X_train, X_test, Y_test)/100.))


def main():
    X,Y = loadData("hw2_lssvm_all.dat.txt")
    X_train = X[:400]
    Y_train = Y[:400]
    X_test  = X[400:]
    Y_test  = Y[400:]

    Ein = np.zeros((len(gamma), len(lamb)))
    Eout = np.zeros((len(gamma), len(lamb)))

    for g in gamma:
        for l in lamb:
            solve(g, l, X_train, Y_train, X_test, Y_test)



if __name__ == '__main__':
    main()