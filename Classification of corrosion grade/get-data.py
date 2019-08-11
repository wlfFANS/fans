import pandas as pd
import numpy as np


def NA():
    data= pd.read_csv('data/1.csv',)
    data .dropna(axis=0, how='any', inplace=True)
    print(data.head())
    print(data.info())
    lack = data.isnull()
    print(lack)
    s = data.isnull().sum()
    print(s)
    # data.to_csv('data/1.csv')



    # data = data.values
    # x, y = np.split(data, (5, ), axis=1)
    # print(x)
    # y = y.ravel()
    # print(y)



    # data.to_csv("data/data-1.csv")
    # data.data =data.values[:,:-1]
    # data.target =data.values[:,-1]
    # print(data.data)
    # print(data.target)


def addl():
    data = pd.read_csv('data/11.csv', )
    # print(data.head())
    data['location'] =data['weld location'] + data['clock location']
    print(data.head())

    data.to_csv('a.csv')

if __name__  =='__main__':
    addl()

