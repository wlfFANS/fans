import lightgbm as lgb
# from sklearn import  LGBMRegressor as lgb
import pandas as pd

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

# 加载数据
print('Load data...')

# iris = load_iris()
# data=iris.data
# target = iris.target
data =pd.read_csv('data/train1.csv')
target =pd.read_csv('data/test1.csv')
X_train,X_test,y_train,y_test =train_test_split(data,target,test_size=0.3)

# df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
# df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')
# y_train = df_train[0].values
# y_test = df_test[0].values
# X_train = df_train.drop(0, axis=1).values
# X_test = df_test.drop(0, axis=1).values

print('Start training...')
# 创建模型，训练模型
gbm = lgb.LGBMRegressor(objective='regression',num_leaves=20,learning_rate=0.000001,n_estimators=20)
gbm.fit(X_train, y_train,eval_set=[(X_test, y_test)],eval_metric='l1',early_stopping_rounds=5)


print('Start predicting...')
# 测试机预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# 模型评估
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)



# feature importances
print('Feature importances:', list(gbm.feature_importances_))

# 网格搜索，参数优化
estimator = lgb.LGBMRegressor(num_leaves=20)

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}

gbm = GridSearchCV(estimator, param_grid)

gbm.fit(X_train, y_train)

print('Best parameters found by grid search are:', gbm.best_params_)


y_pred = y_pred.astype(int) #转化成整型
print ('混淆矩阵---->\n',confusion_matrix(y_test,y_pred))
y_test =(y_test.values.ravel().astype('float64'))





print("accuracy_score:", accuracy_score(y_test, y_pred))
print("precision_score:", metrics.precision_score(y_test, y_pred,average='weighted'))
print("recall_score:", metrics.recall_score(y_test, y_pred,average='weighted'))
print("f1_score:", metrics.f1_score(y_test,y_pred, average='weighted'))
print("f0.5_score:", metrics.fbeta_score(y_test, y_pred, beta=0.5,average='weighted'))
print("f2_score:", metrics.fbeta_score(y_test, y_pred, beta=2.0,average='weighted'))
