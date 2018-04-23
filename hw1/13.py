import sys
sys.path.insert(0, "/Users/goodhat/Downloads/libsvm-3.22/python")

import numpy as np
import matplotlib.pyplot as plt
logC = [-5, -3, -1, 1, 3]





def main():

    # Load model from Question 12
    figureList = []
    for e in logC:
        print ("C: " + str(10**e))
        totalSV = open("model_12_"+str(e), 'r').read().split('\n')[6].split(' ')[1]
        print(totalSV)
        figureList.append(float(totalSV))

    figure = plt.figure()
    figure.suptitle("Number of SVs in different c",fontsize=20)
    plt.xlabel('C',fontsize = 18)
    plt.ylabel('Number of SVs', fontsize = 16)
    plt.plot([pow(10,C) for C in logC], figureList)
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':
    main()
