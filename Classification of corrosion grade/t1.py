#划分训练集合测试集
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
import warnings
from sklearn import  metrics
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)



data =pd.read_csv('data/data-1.csv',header=0,)
X = data.iloc[:,0:4]
y = np.ravel(data.iloc[:,4]) #降成一维，类似np.flatten(),但是np.flatten是拷贝，而ravel是引用
print(X)
print(y)
#随机划分训练集和测试集
#test_size:测试集占比
#random_state:随机种子，在需要重复试验的时候，保证得到一组一样的随机数。
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=32)


#标准化数据


scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)


#使用keras模型化数据
model = Sequential()
#添加输入层
model.add(Dense(25,activation='relu',
               input_shape=(4,)))
#添加隐藏层
model.add(Dense(8,activation='relu'))

#添加输出层
model.add(Dense(1,activation='sigmoid'))
print(model.output_shape)
print(model.summary())
print (model.get_weights())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #verbose = 1 查看输出过程
model.fit(X_train,y_train,epochs=30,batch_size=100,verbose=1)

#预测结果
y_pred = model.predict(X_test)
print (y_pred[:5])
print(y_test[:5])


#模型评估
score = model.evaluate(X_test,y_test,verbose=1)

#socre的两个值分别代表损失(loss)和精准度(accuracy)
print (score)


#统计Precision、Recall、F1值
y_pred = y_pred.astype(int) #转化成整型
print ('混淆矩阵---->\n',confusion_matrix(y_test,y_pred))

#precision
precision = precision_score(y_test,y_pred,average='weighted')
print ('精度--->\n',precision)

#Recall
recall = recall_score(y_test,y_pred,average='weighted')
print ('召回率--->\n',recall)

#F1 score
f1 = f1_score(y_test,y_pred,average='weighted')
print ('F1----->\n',f1)

print(metrics.accuracy_score(y_test,y_pred,))