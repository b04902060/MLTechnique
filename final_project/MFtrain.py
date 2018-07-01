import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


# parameters
K = 30
ITERAIONS = 100
T = 1
alpha = 0.001


# read file
Users = np.load('data/users.npy').item()
userNum = len(Users)
Books = np.load('data/books.npy').item()
bookNum = len(Books)
RatingData = np.load('data/ratingData.npy')
TestData = np.load('data/testData.npy')

# model
targetMatrix = np.zeros((userNum, bookNum))
userMatrix = np.random.rand(userNum, K)
bookMatrix = np.random.rand(bookNum, K)
RatingData_val = RatingData[-1000:, :]
RatingData_train = RatingData[:-1000,:]

def predict(X, userMatrix, bookMatrix):
    num = X.shape[0]
    prediction = np.zeros((num,1), dtype=int)
    for i in range(num):
        userIndex = X[i][0]
        bookIndex = X[i][1]
        if(bookIndex == -1): # The ISBN of this book is not in the data set.
            prediction[i][0] = 7 # just guess 7
        else:
            prediction[i][0] = round(userMatrix[userIndex].dot(bookMatrix[bookIndex]))

        if(prediction[i][0] > 10):
            prediction[i][0] = 0
        if(prediction[i][0] <= 0):
            prediction[i][0] = 1
    return prediction

def evaluation(Y_pred, Y):
    num = Y_pred.shape[0]
    residual = np.absolute(Y_pred-Y)
    err = np.sum(residual)/float(num)
    return err

r_pred_train = np.zeros((RatingData_train.shape[0],1), dtype=int)
r_pred_val = np.zeros((RatingData_val.shape[0],1), dtype=int)

for t in range(ITERAIONS):
    print("Iteration: ", t)
    np.random.shuffle(RatingData_train)
    for rating in RatingData_train:
        userIndex = rating[0]
        bookIndex = rating[1]
        if (bookIndex == -1):
            continue

        userFeatureVector = userMatrix[userIndex]
        bookFeatureVector = bookMatrix[bookIndex]

        # gradient descent
        r_ans = float(rating[2])
        r_pred = userFeatureVector.dot(bookFeatureVector)
        err = r_ans - r_pred
        if(t%2==0):
            userFeatureVector = userFeatureVector + 2*alpha*err* bookFeatureVector
            userMatrix[userIndex] = userFeatureVector
        else:
            bookFeatureVector = bookFeatureVector + 2*alpha*err* userFeatureVector
            bookMatrix[bookIndex] = bookFeatureVector

    # Uncomment to gernerate model in each iteration.
    # if(t % 5 == 0):
    #     path = 'models/10_3/'#+str(K)+'_'+str(T)+'_'+str(alpha)+'/'
    #     np.save(path+'userMatrix'+str(t)+'.npy', userMatrix)
    #     np.save(path+'bookMatrix'+str(t)+'.npy', bookMatrix)




#save models
np.save('models/userMatrix.npy', userMatrix)
np.save('models/bookMatrix.npy', bookMatrix)
