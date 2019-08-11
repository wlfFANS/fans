from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
# from sklearn.metrics import accuracy
import numpy as np
from sklearn.model_selection import GridSearchCV
# from sklearn.grid_search import GridSearchCV
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, fbeta_score




data = pd.read_csv('data/data-1.csv', )
train = data.iloc[:, 0:4]
target = data.iloc[:, 4]
# target=target.ravel()

# train =pd.read_csv('data/train.csv',header=0,)
# target =pd.read_csv('data/test.csv')
train  =train.values
target =target.values
# target[target == 0] = -1
target= target.ravel()




X_train,X_test,y_train,y_test = train_test_split(train,target,test_size=0.3,random_state=30,stratify=target)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)


model=svm.SVC(C=10, kernel='rbf',gamma=0.005 )
# model =svm.SVC(C=1, kernel='linear', class_weight={0: 1, 1: 50})
# model = svm.SVC(C=8, kernel='rbf', gamma=0.5, class_weight={0: 1, 1: 2})
# model= svm.SVC(C=8, kernel='rbf', gamma=0.5, class_weight={0: 1, 1: 10})



# parameters={'kernel':['linear','rbf','sigmoid',],'C':np.linspace(0.1,20,50),'gamma':np.linspace(0.01,2,20)}
# svc = svm.SVC()
# model = GridSearchCV(svc,parameters,cv=3,scoring='accuracy')
#


model.fit(X_train,y_train)
# cv_result = pd.DataFrame.from_dict( model.cv_results_ )

# print(model.best_params_)
acc =model.score(X_test,y_test)
# print(acc)
print('正确率：%.6f%%' % (100 * float(acc)))

y_pred = model.predict(X_test)
print('正确率：\t', accuracy_score(y_test, y_pred))
print(classification_report(y_true=y_test, y_pred=y_pred))







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



# y_pred =pd.DataFrame(y_pred)
# y_pred.to_csv('data/d.csv')
# y_test=pd.DataFrame(y_test)
# y_test.to_csv('data/t.csv')