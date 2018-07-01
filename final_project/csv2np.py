import pandas as pd
import numpy as np

PATH = '/Users/goodhat/Downloads/data/'

with open(PATH+'books.csv', encoding='utf-8') as f:
    BooksData = pd.read_csv(f).values

with open(PATH+'users.csv', encoding='utf-8') as f:
    UsersData = pd.read_csv(f).values

with open(PATH+'book_ratings_train.csv', encoding='utf-8') as f:
    RatingData = pd.read_csv(f).values

with open(PATH+'book_ratings_test.csv', encoding='utf-8') as f:
        TestData = pd.read_csv(f).values


Users = dict()
for i in range(len(UsersData)):
    Users[UsersData[i][0]] = (i, UsersData[i][1], UsersData[i][2])

np.save('data/users.npy', Users)


Books = dict()
for i in range(len(BooksData)):
    # preprocess the ISBN
    # if (BooksData[i][0][-1] == 'x'): # cut postfix x
    #      BooksData[i][0] = BooksData[i][0][:-1]
    # if (BooksData[i][0][-1] == 'X'): # cut postfix x
    #      BooksData[i][0] = BooksData[i][0][:-1]
    # try: # cut prefix 0
    #     BooksData[i][0] = int(BooksData[i][0])
    # except:
    #     pass

    Books[BooksData[i][0]] = (i, BooksData[i][1], BooksData[i][2], BooksData[i][3])


# change user_id and ISBN into my own index
for i in range(len(RatingData)):
    # user_id
    RatingData[i][0] = Users[RatingData[i][0]][0]

    # # book ISBN
    # if(RatingData[i][1][-1] == 'X'):
    #     RatingData[i][1] = RatingData[i][1][:-1]
    # if(RatingData[i][1][-1] == 'x'):
    #     RatingData[i][1] = RatingData[i][1][:-1]
    # try:
    #     RatingData[i][1] = int(RatingData[i][1])
    # except:
    #     pass
    try:
        RatingData[i][1] = Books[RatingData[i][1]][0]
    except:
        RatingData[i][1] = -1


for i in range(len(TestData)):
    TestData[i][0] = Users[TestData[i][0]][0]
    # if(TestData[i][1][-1] == 'X'):
    #     TestData[i][1] = TestData[i][1][:-1]
    # if(TestData[i][1][-1] == 'x'):
    #     TestData[i][1] = TestData[i][1][:-1]
    # try:
    #     TestData[i][1] = int(TestData[i][1])
    # except:
    #     pass
    try:
        TestData[i][1] = Books[TestData[i][1]][0]
    except:
        TestData[i][1] = -1

np.save('data/books.npy', Books)
np.save('data/ratingData.npy', RatingData)
np.save('data/testData.npy', TestData)
print(len(TestData))
