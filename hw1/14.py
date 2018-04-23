import sys
#import LIBSVM
#sys.path.insert(0, "/Users/goodhat/Downloads/libsvm-3.22/python")

from svmutil import *
import numpy as np
import csv
import matplotlib.pyplot as plt
from math import sqrt, exp


training_set_file = "/Users/goodhat/Downloads/features.train.txt"
testing_set_file = "/Users/goodhat/Downloads/features.test.txt"
TARGET = 0
logC = [-3, -2, -1, 0, 1]


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
    m = svm_train(Y, X, '-h 0 -t 2 -g 80 -q -c ' + str(C))
    return m

def kernel(x,y):
    a = (x[0]-y[0])**2+(x[1]-y[1])**2
    return exp(-80*a)

def main():
    X_train, Y_train = read_data(training_set_file, TARGET)
    X_test, Y_test = read_data(testing_set_file, TARGET)
    for e in logC:
        print ("C: " + str(10**e))
        model = train(X_train, Y_train, 10**e)
        svm_save_model("model_14_"+str(e), model)
        #print ("C: " + str(10**e) + " Ein: ")
        #p_label, p_acc, p_val = svm_predict(Y_train, X_train, model) # Ein
    figureList = []
    for e in logC:
        result = 0
        data = open("model_14_"+str(e), 'r').read().split('\n')[9:-1]
        data = [(float(num.split(' ')[0]), float(num.split(' ')[1].split(':')[1]), float(num.split(' ')[2].split(':')[1])) for num in data]
        for i in range(len(data)):
            for j in range(len(data)):
                result += data[i][0]*data[j][0]*kernel(data[i][1:],data[j][1:])
        result = 1/sqrt(result)
        figureList.append(result)


    figure = plt.figure()
    figure.suptitle("Distance to hyperplane",fontsize=20)
    plt.xlabel('C',fontsize = 18)
    plt.ylabel('distance', fontsize = 16)
    plt.plot([pow(10,C) for C in logC], figureList)
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':
    main()
