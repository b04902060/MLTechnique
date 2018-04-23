import sys
#import LIBSVM
# sys.path.insert(0, "/Users/goodhat/Downloads/libsvm-3.22/python")

from svmutil import *
import numpy as np
import csv
import matplotlib.pyplot as plt

training_set_file = "/Users/goodhat/Downloads/features.train.txt"
testing_set_file = "/Users/goodhat/Downloads/features.test.txt"
TARGET = 0 # TARGET = 1, not TARGET = 0
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
    return svm_train(Y, X, '-t 0 -q -h 0 -c ' + str(C))

def main():

    # Read training data and train models
    # When C = 1000, it won't stop utill it reaches iteration limit
    X_train, Y_train = read_data(training_set_file, TARGET)
    X_test, Y_test = read_data(testing_set_file, TARGET)
    for e in logC:
        print ("C: " + str(10**e))
        model = train(X_train, Y_train, 10**e)
        svm_save_model("model_11_"+str(e), model)



    # Compute w and draw figure
    model_name = ""
    w = [0., 0.]
    figureList = []
    for e in logC:
        model_name = "model_11_"+str(e)
        f = open(model_name,'r').read().split('\n')[8:-1]
        for line in f:
            alphay = float(line.split(' ')[0])
            x1 = float(line.split(' ')[1].split(':')[1])
            x2 = float(line.split(' ')[2].split(':')[1])
            w[0] += alphay * x1
            w[1] += alphay * x2
        figureList.append(w[0]**2 + w[1]**2)
    figure = plt.figure()
    figure.suptitle("|W| in different c",fontsize=20)
    plt.xlabel('C',fontsize = 18)
    plt.ylabel('|W|', fontsize = 16)
    plt.plot([pow(10,C) for C in logC], figureList, lw=2)
    plt.xscale('log')
    plt.show()

if __name__ == '__main__':
    main()
