import numpy as np
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split,RepeatedKFold
from sklearn.pipeline import Pipeline
from DBN import UnsupervisedDBN
# from UnsupervisedDBNTensorflow import BinaryRBM,UnsupervisedDBN
from datetime import datetime
from sklearn import preprocessing
from sklearn import svm
import pandas as pd
from numpy import float64
#grid search -->sklearn
from sklearn.model_selection import GridSearchCV

#读取CSV文件。header = None 表示这个矩阵没有指定行索引，程序自动添加
data = pd.read_csv('train.csv',header=None)
test_data = pd.read_csv('test.csv',header=None)

#将CSV文件转换为矩阵
Data_matrix = data.as_matrix()
# print('Data\'s matrix shape: ',Data_matrix.shape)
# for i in Data_matrix:
# 	print(i)
Test_matrix = test_data.as_matrix()
print('TestData\'s matrix shape: ',Test_matrix.shape)

#提取矩阵中的标签列,作为标签矩阵使用
Table_matrix = Data_matrix[:,4]
#删除矩阵中标签列,训练矩阵
Data_matrix = np.delete(Data_matrix,4,axis=1)

#删除测试数据第一行无用行
Test_matrix = np.delete(Test_matrix,0,axis=0)
print('删除第一行后测试数据矩阵：\n',Test_matrix.shape,'\n',Test_matrix)
#测试数据标签矩阵
Test_table_matrix = Test_matrix[:,4]
print('测试数据标签矩阵：\n',Test_table_matrix.shape,'\n',Test_table_matrix)
#删除标签列后，作为测试数据矩阵
Test_matrix = np.delete(Test_matrix,4,axis=1)
print('测试数据矩阵：\n',Test_matrix.shape,'\n',Test_matrix)

# print(Table_matrix)
#数据分割，交叉验证 -->>K-fold循环
# Data_Train,Data_Test,Table_Train,Table_Test = train_test_split(Data_matrix,Table_matrix,
# 															   test_size=0.2,
# 															   random_state=0)
# print('Training data	Test data	TrainTable	TestTable')
# print(Data_Train.shape,'	',Data_Test.shape,''	,Table_Train.shape,'	',Table_Test.shape)
# print('#############################################################################################')
# print('Training:',Data_Train.shape)
# print('Test:',Data_Test.shape)
# print('TrainTable:',Table_Train.shape)
# print('TestTable:',Table_Test.shape)
#数据归一化
'''
下一步
1、要将矩阵进行密集化，当前的矩阵0太多，为稀疏矩阵
2、先尝试用KDD的Train数据集、Test数据集，不进行数据分割
3、再尝试如何对两个矩阵进行分割


将属性缩放到一个指定的最大和最小值（通常是1-0）之间，这可以通过preprocessing.MinMaxScaler类实现。

使用这种方法的目的包括：

1、对于方差非常小的属性可以增强其稳定性。

2、维持稀疏矩阵中为0的条目。
'''
print('###############MinMaxScaler######################')
min_max_scaler = preprocessing.MinMaxScaler()
Data_matrix_minmax = min_max_scaler.fit_transform(Data_matrix)
Data_matrix_minmax = Data_matrix_minmax.astype(float64)
print(Data_matrix)
print()
Test_matrix_minmax = min_max_scaler.transform(Test_matrix)
print(Test_matrix_minmax)
#配置DBN和分类器
'''
SVM: kernel = sigmoid,gamma = ?,C = ?

DBN:hidden_layers_structure=[512,42],
batch_size=10,
learning_rate_rbm = 0.06,
n_epochs_rbm = 20,(进行训练的迭代次数)
activation_function='sigmoid'

Best Parameters:
{'dbn__learning_rate_rbm': 0.007, 'clf_svm__kernel': 'rbf', 'clf_svm__C': 1000, 'dbn__activation_function': 'relu', 'clf_svm__gamma': 0.008}
'''

clf_svm = svm.SVC(kernel='rbf',gamma = 0.01,C = 10000)
# clf_svm.gamma =
# logistic = linear_model.LogisticRegression()
#[512,128,42]  0.16      0.13      0.14      5039
dbn = UnsupervisedDBN(hidden_layers_structure=[64,32,4],
					  batch_size=20,
					  learning_rate_rbm=0.1,
					  n_epochs_rbm=25,
					  activation_function='relu')
#组合DBN和SVM分类器
# classifier = Pipeline(steps=[('dbn',dbn),('logistic',logistic)])
classifier = Pipeline(steps=[('dbn',dbn),('clf_svm',clf_svm)])
a  = datetime.now()
#进行fit
# logistic.C = 6000.0
classifier.fit(Data_matrix_minmax,Table_matrix)
# logistic_classifier = linear_model.LogisticRegression(C=100.0)
# logistic_classifier.fit(Data_Train_minmax, Table_Train)
clf_svm.kernel = 'rbf'
clf_svm.gamma = 'auto'
clf_svm.fit(Data_matrix_minmax,Table_matrix)
print()


print("SVC using RBM features:\n%s\n" % (metrics.classification_report(Test_table_matrix, classifier.predict(Test_matrix_minmax))))
# print("LinearSVC:\n%s\n" % (metrics.classification_report(Table_Test, logistic_classifier.predict(Data_Test_minmax))))
print("SVC:\n%s\n" % (metrics.classification_report(Test_table_matrix, clf_svm.predict(Test_matrix_minmax))))
print("accuracy_score:", (metrics.accuracy_score(Test_table_matrix, clf_svm.predict(Test_matrix_minmax))))
b = datetime.now()
err_sum = 0
# X = np.concatenate((Data_Train_minmax,Data_Test_minmax),axis=0)
for x_error in dbn.reconstruction_accuracy(Test_matrix_minmax):
		err_sum = err_sum + x_error
	# -------------------
print('重构误差：',err_sum/len(Test_matrix_minmax))
print('重构误差和误差的乘积：',
		  err_sum / len(Test_matrix_minmax) * (1 - metrics.precision_score(Test_table_matrix, classifier.predict(Test_matrix_minmax), average='weighted')))
# dbn.hidden_layers_structure.insert(0,i)
print('Hidden layers structure: ',dbn.hidden_layers_structure)
print('計算時間：',(b-a))

# random_state = 12883823
# rkf = RepeatedKFold(n_splits=10,n_repeats=2,random_state=random_state)
# for train,test in rkf.split(Data_matrix):
# 	print(train.shape)
# 	print()
# 	print(test.shape)

