import csv
from math import sqrt, log
import numpy as np
import scipy.linalg as lin
import pandas as pd
import matplotlib.pyplot as plt


def loadData(filename):
    data = pd.read_csv(filename, sep=' ', header=None)
    data = data.values
    col, row = data.shape
    X = data[:, 0: row-1]
    Y = data[:, row-1:row]
    return X, Y

def decision_stump(X, Y, interval, U):
    row, col = X.shape
    r, c = interval.shape
    err_best = 1
    theta = 0
    s = 0
    dim = 0
    for i in range(col):
        # Use a matrix to store the prediction of each threshold
        Y_predict_matrix = np.sign(np.tile(X[:, i:i+1], (1, r)).T-interval[:, i:i+1]).T

        # record the error of each prediction
        err_positive = (Y_predict_matrix!=Y).T.dot(U)
        err_negative = (-1*Y_predict_matrix!=Y).T.dot(U)

        # get the dimension with the smallest error
        if np.min(err_positive) < np.min(err_negative):
            if np.min(err_positive) < err_best:
                s = 1
                err_best = np.min(err_positive)
                dim = i
                theta = interval[np.argmin(err_positive), i]
        else:
            if np.min(err_negative) < err_best:
                s = -1
                err_best = np.min(err_negative)
                dim = i
                theta = interval[np.argmin(err_negative), i]
    return err_best, theta, s, dim

def ada_boost(X, Y, T, pbm):
    row, col = X.shape
    U = np.ones((row, 1))/row
    X_sort = np.sort(X, 0)

    # get all interval of X in every dimension
    interval = (np.r_[X_sort[0:1, :] - 0.1, X_sort[0:row - 1, :]] + X_sort) / 2

    # hypothesis gs
    theta = np.zeros((T,))
    s = np.zeros((T,))
    dim = np.zeros((T,)).astype(int)
    alpha = np.zeros((T,))

    # error of g(i) to compute its weight
    err = np.zeros((T,))
    Ut = []
    # iteration
    for i in range(T):
        Ut.append(np.sum(U))
        # get a new weak hypothesis g by new weighted U
        err[i], theta[i], s[i], dim[i] = decision_stump(X, Y, interval, U)

        # compute the error rate of g
        yhat = s[i]*np.sign(X[:, dim[i]]-theta[i])
        weight = np.sqrt((1-err[i])/err[i])

        # get new weight and normalize
        U = [ U[i]/weight  if Y[i]==yhat[i] else U[i]*weight for i in range(len(U))]

        # for problem 14
        if pbm != "14":
            U /= np.sum(U)

        # record the voting weight of this g
        alpha[i] = np.log(weight)
    return theta, dim, s, alpha, Ut

def predict(X, theta, dim, s, alpha):
    row, col = X.shape
    g_num = len(theta)
    Y_hat = [0 for i in range(row)]
    for i in range(g_num):
        for j in range(row):
            if float(X[j][dim[i]]) > float(theta[i]):
                Y_hat[j] +=  s[i] * alpha[i]
            else:
                Y_hat[j] += -s[i] * alpha[i]

    Y_hat = [np.sign(y) for y in Y_hat]
    return np.array(Y_hat).reshape((row, 1))

def main():
    X_train, Y_train = loadData("hw3_train.dat.txt")
    X_test, Y_test = loadData("hw3_test.dat.txt")
    T = 300
    problem11(X_train, Y_train, X_test, Y_test, T)
    problem13(X_train, Y_train, X_test, Y_test, T)
    problem14(X_train, Y_train, X_test, Y_test, T)
    problem15(X_train, Y_train, X_test, Y_test, T)
    problem16(X_train, Y_train, X_test, Y_test, T)


# Plot a figure for t versus Ein(gt). What is Ein(g1) and what is α1?
def problem11(X_train, Y_train, X_test, Y_test, T):
    r_train, c_train = X_train.shape
    r_test, c_test = X_test.shape

    Ein = []
    theta, dim, s, alpha, Ut = ada_boost(X_train, Y_train, T, "11")
    for i in range(T):
        Y_pred = predict(X_train, [theta[i]], [dim[i]], [s[i]], [alpha[i]])
        Ein.append(np.sum(Y_pred!=Y_train)/float(r_train))
    print('Ein(g1)', Ein[0], '\nalpha1:', alpha[0])
    fig = plt.figure()
    plt.plot(np.arange(300), Ein)
    fig.savefig('11_Eing_per_t')
    fig.clf()

# Plot a figure for t versus Ein(Gt). G = g0~gt aggregating.
def problem13(X_train, Y_train, X_test, Y_test, T):
    r_train, c_train = X_train.shape
    r_test, c_test = X_test.shape

    Ein = []
    theta, dim, s, alpha, Ut= ada_boost(X_train, Y_train, T, "13")
    for i in range(T):
        Y_pred = predict(X_train, theta[0:i+1], dim[0:i+1], s[0:i+1], alpha[0:i+1])
        Ein.append(np.sum(Y_pred!=Y_train)/float(r_train))
    print('Ein(G)', Ein[T-1])
    fig = plt.figure()
    plt.plot(np.arange(300), Ein)
    fig.savefig('13_EinG_per_t')
    fig.clf()


def problem14(X_train, Y_train, X_test, Y_test, T):
    r_train, c_train = X_train.shape
    r_test, c_test = X_test.shape

    Ein = []
    theta, dim, s, alpha, Ut = ada_boost(X_train, Y_train, T, "14")
    print("U2: ", Ut[1])
    print("UT: ", Ut)
    fig = plt.figure()
    plt.plot(np.arange(T), Ut)
    fig.savefig('14_Ut')
    fig.clf()

def problem15(X_train, Y_train, X_test, Y_test, T):
    r_train, c_train = X_train.shape
    r_test, c_test = X_test.shape

    Eout = []
    theta, dim, s, alpha, Ut = ada_boost(X_train, Y_train, T, "15")
    for i in range(T):
        Y_pred = predict(X_test, [theta[i]], [dim[i]], [s[i]], [alpha[i]])
        Eout.append(np.sum(Y_pred!=Y_test)/float(r_test))
    print("Eout(g1): ", Eout[0])
    fig = plt.figure()
    plt.plot(np.arange(T), Eout)
    fig.savefig('15_Eoutg_per_t')
    fig.clf()

def problem16(X_train, Y_train, X_test, Y_test, T):
    r_train, c_train = X_train.shape
    r_test, c_test = X_test.shape

    Eout = []
    theta, dim, s, alpha, Ut = ada_boost(X_train, Y_train, T, "15")
    for i in range(T):
        # This could be improved by getting each Eout in one prediction
        Y_pred = predict(X_test, theta[0:i+1], dim[0:i+1], s[0:i+1], alpha[0:i+1])
        Eout.append(np.sum(Y_pred!=Y_test)/float(r_test))
    print("Eout(G): ", Eout[-1])
    fig = plt.figure()
    plt.plot(np.arange(T), Eout)
    fig.savefig('16_EoutG_per_t')
    fig.clf()


if __name__ == '__main__':
    main()
