# https://www.cnblogs.com/wj-1314/p/9591369.html
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# load dataset

dataframe = pd.read_csv("data/data-1.csv",header=0)

dataset = dataframe.values

X = dataset[:, 0:4]

Y = dataset[:,4]

# encode class values as integers
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
# convert integers to dummy variables (one hot encoding)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y)

# define model structure
def baseline_model():
    model = Sequential()
    model.add(Dense(output_dim=100, input_dim=4, activation='relu'))

    model.add(Dense(output_dim=50, input_dim=100, activation='relu'))
    #
    # model.add(Dense(output_dim=16, input_dim=32, activation='relu'))

    model.add(Dense(output_dim=5, input_dim=50, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=1000, batch_size=10,verbose=1)
# splitting data into training set and test set. If random_state is set to an integer, the split datasets are fixed.
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3, random_state=10)
estimator.fit(X_train, Y_train)

# make predictions
pred = estimator.predict(X_test)

# inverse numeric variables to initial categorical labels
init_lables = encoder.inverse_transform(pred)
print(init_lables)
# k-fold cross-validate
seed = 42
np.random.seed(seed)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))