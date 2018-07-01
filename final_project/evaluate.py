import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

TestData = np.load('data/testData.npy')
RatingData = np.load('data/ratingData.npy')


X = RatingData[:,0:2]
Y = RatingData[:, -1].reshape(X.shape[0],1)

X_val = RatingData[-1000:, 0:2]
X_train = RatingData[:-1000, 0:2]
Y_val = RatingData[-1000:, -1].reshape(X_val.shape[0], 1)
Y_train = RatingData[:-1000, -1].reshape(X_train.shape[0], 1)

# To computer Eval versus T, and plot a figure, uncomment these.
# X_test = TestData[:,0:2]
# Ein = []
# Eval= []
# path = 'models/10_3/'
# for t in range(40):
#     print(t)
#     userMatrix = np.load(path+'userMatrix'+str(t*5)+'.npy')
#     bookMatrix = np.load(path+'bookMatrix'+str(t*5)+'.npy')
#
#     Y_pred = predict(X_train, userMatrix, bookMatrix)
#     err = evaluation(Y_pred, Y_train)
#     Ein.append(err)
#
#     Y_pred = predict(X_val, userMatrix, bookMatrix)
#     err = evaluation(Y_pred, Y_val)
#     Eval.append(err)
#
# print(Ein)
# print(Eval)
#
# fig = plt.figure()
# plt.plot(np.arange(40)*5, Ein, 'b-', label="Ein")
# plt.plot(np.arange(40)*5, Eval, 'r-', label="Eval")
# plt.ylabel('Error(task 1)')
# plt.xlabel('Iteration')
# plt.legend()
# plt.title('K=30 tango')
# fig.savefig('figure/30_3_tango')
# fig.clf() [2.2276432, 1.6013413, 1.38293424, 1.76232484, 2.2544231]

# generate prediction of test data as csv
X_test = np.load('data/testData.npy')
userMatrix = np.load('models/userMatrix.npy')
bookMatrix = np.load('models/bookMatrix.npy')
Y_pred = predict(X_test, userMatrix, bookMatrix)
Output = pd.DataFrame(Y_pred, columns=['Rating'])
Output.to_csv('30_random.csv', index=False, header=False)
