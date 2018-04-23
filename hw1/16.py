import sys
#import LIBSVM
#sys.path.insert(0, "/Users/goodhat/Downloads/libsvm-3.22/python")

from svmutil import *
import numpy as np
import csv
import matplotlib.pyplot as plt
from random import shuffle


training_set_file = "/Users/goodhat/Downloads/features.train.txt"
testing_set_file = "/Users/goodhat/Downloads/features.test.txt"
TARGET = 0
logGamma = [-1, 0, 1, 2, 3]


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

def train(X, Y, gamma):
    return svm_train(Y, X, '-t 2 -q -c 0.1 -g ' + str(gamma))


def shuffle_data(X, Y):
    data = list(zip(X, Y))
    shuffle(data)
    return zip(*data)
def main():
    X_all, Y_all = read_data(training_set_file, TARGET)


    figureList = [0,0,0,0,0]
    for i in range(100):
        X_all, Y_all = shuffle_data(X_all, Y_all)

        bestGamma = -3
        bestEval = 0
        print("======= Iteration " + str(i+1) + " =======")
        for e in logGamma:
            print ("Gamma: " + str(10**e))
            model = train(X_all[1000:], Y_all[1000:], 10**e)
            #svm_save_model("model_15_"+str(e), model)
            #model = svm_load_model("model_15_"+str(e))
            #print ("C: " + str(10**e) + " Ein: ")
            p_label, p_acc, p_val = svm_predict(Y_all[:1000], X_all[:1000], model) # Ein
            if p_acc[0] > bestEval:
                bestGamma = e
                bestEval = p_acc[0]
        figureList[bestGamma + 1] += 1
        #print("Eout:"+ str(p_acc[0]))

    figure = plt.figure()
    figure.suptitle("Validation (C = 0.1)",fontsize=20)
    plt.xlabel('gamma',fontsize = 18)
    plt.ylabel('number of times', fontsize = 16)
    plt.plot([pow(10,g) for g in logGamma], figureList)
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':
    main()
