import sys
# import LIBSVM
#sys.path.insert(0, "/Users/goodhat/Downloads/libsvm-3.22/python")

from svmutil import *
import numpy as np
import csv
import matplotlib.pyplot as plt


training_set_file = "/Users/goodhat/Downloads/features.train.txt"
testing_set_file = "/Users/goodhat/Downloads/features.test.txt"
TARGET = 0
logGamma = [0, 1, 2, 3, 4]


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
    m = svm_train(Y, X, '-h 0 -t 2 -q -c 0.1 -g ' + str(gamma))
    return m

def main():
    X_train, Y_train = read_data(training_set_file, TARGET)
    X_test, Y_test = read_data(testing_set_file, TARGET)
    figureList = []
    for e in logGamma:
        print ("Gamma: " + str(10**e))
        model = train(X_train, Y_train, 10**e)
        svm_save_model("model_15_"+str(e), model)
        model = svm_load_model("model_15_"+str(e))
        #print ("C: " + str(10**e) + " Ein: ")
        p_label, p_acc, p_val = svm_predict(Y_test, X_test, model) # Ein
        figureList.append(100. - p_acc[0])
        #print("Eout:"+ str(p_acc[0]))

    figure = plt.figure()
    figure.suptitle("Eout in different gamma (C = 0.1)",fontsize=20)
    plt.xlabel('gamma',fontsize = 18)
    plt.ylabel('Eout (100-accuracy)%', fontsize = 16)
    plt.plot([pow(10,g) for g in logGamma], figureList)
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':
    main()
