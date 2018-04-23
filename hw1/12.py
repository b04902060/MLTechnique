import sys
#import LIBSVM
# sys.path.insert(0, "/Users/goodhat/Downloads/libsvm-3.22/python")

from svmutil import *
import numpy as np
import csv
import matplotlib.pyplot as plt


training_set_file = "/Users/goodhat/Downloads/features.train.txt"
testing_set_file = "/Users/goodhat/Downloads/features.test.txt"
TARGET = 8
logC = [-5, -3, -1, 1, 3]


def read_data(filepath, target=0):
    X = []
    Y = []
    file = csv.reader(open(filepath, "r"), delimiter = ' ')
    for r in file:
        X.append([float(r[6]), float(r[8])])
        Y.append([r[3]])

    set_label(target, Y) # somehow python pass reference to function
    return X, Y


def set_label(target, Y):
    for i in range(len(Y)):
        if eval(Y[i][0]) == target:
            Y[i] = 1
        else:
            Y[i] = 0
    return


def train(X, Y, C):
    m = svm_train(Y, X, '-t 1 -d 2 -q -c ' + str(C))
    return m


def main():
    X_train, Y_train = read_data(training_set_file, TARGET)
    X_test, Y_test = read_data(testing_set_file, TARGET)
    figureList = []
    for e in logC:
        print ("C: " + str(10**e))
        model = train(X_train, Y_train, 10**e)
        svm_save_model("model_12_"+str(e), model)
        model = svm_load_model("model_12_"+str(e))
        p_label, p_acc, p_val = svm_predict(Y_train, X_train, model) # Ein
        figureList.append(p_acc[0])


    # Plot the figure
    figure = plt.figure()
    figure.suptitle("Ein in different c",fontsize=20)
    plt.xlabel('C',fontsize = 18)
    plt.ylabel('Ein', fontsize = 16)
    plt.plot([pow(10,C) for C in logC], figureList, lw=2)
    plt.xscale('log')
    plt.show()

if __name__ == '__main__':
    main()
