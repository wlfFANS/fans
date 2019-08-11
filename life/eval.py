
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


# 计算RMSE
def calcRMSE(true,pred):
    return np.sqrt(mean_squared_error(true, pred))


# 计算MAE
def calcMAE(true,pred):
    return mean_absolute_error(true, pred)


# 计算MAPE
def calcMAPE(true, pred):


    return np.sum(np.abs((true-pred)/true))/len(true)


# 计算SMAPE
def calcSMAPE(true, pred):
    delim = (np.abs(true)+np.abs(pred))/2.0

    a = np.sum((np.abs(pred-true)/delim))/true.shape[0]
    return a