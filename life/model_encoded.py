import tensorflow as tf
from keras.models import Model
import keras.layers as kl
import keras as kr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import eval




def nnmodel(epochs, regularizer1, regularizer2):

    train_data = np.array(pd.read_csv("feat/autoencoded_train_data.csv", index_col=0).fillna(0))

    # length = len(train_data)
    train_data = np.reshape(train_data, (len(train_data), 20))
    print(np.shape(train_data))
    test_data = np.array(pd.read_csv("feat/autoencoded_test_data.csv", index_col=0).fillna(0))

    test_data = np.reshape(test_data, (len(test_data), 20))
    train_y = np.array(pd.read_csv("feat/autoencoded_train_y.csv", index_col=0).fillna(0))

    test_y = np.array(pd.read_csv("feat/autoencoded_test_y.csv", index_col=0).fillna(0))

    pr = np.array(pd.read_csv("get_data/new_data_test.csv", index_col=0).fillna(0))



    model = kr.models.Sequential()
    # model.add(kl.Dense(50, activation="sigmoid", activity_regularizer=kr.regularizers.l2(0)))
    model.add(kl.Dense(8, input_dim=20, activation="relu", activity_regularizer=kr.regularizers.l2(regularizer1)))
    model.add(kl.Dense(8, activation="relu", activity_regularizer=kr.regularizers.l2(regularizer2)))
    # model.add(kl.Dense(100))
    model.add(kl.Dense(1))

    model.compile(optimizer="sgd", loss="mean_squared_error")
    model.fit(train_data, train_y, epochs=epochs)
    # model.save("models/final_model.h5")



    predicted_data = []
    predicted_pr = []
    for i in range(len(test_data)):
        test_data[i] =test_data[i].reshape(1,20)
        prediction = model.predict(np.reshape(test_data[i], (1, 20)))
        prediction = pd.DataFrame(prediction).fillna(0).values
        predicted_data.append(prediction)
        pre = np.around(pr[i], decimals=1)

        p_pred = np.exp(prediction)*(pre-0.1)
        predicted_pr.append(p_pred)
        # print(test_data[i])

    # print(model.evaluate(test_data, test_y))
    pd.DataFrame(np.reshape(predicted_pr, (len(predicted_pr, ))))

    predicted_pr =np.reshape(predicted_pr, (len(predicted_pr)))
    predicted_data = np.reshape(predicted_data, (len(predicted_data)))
    p_r_score = r2_score(np.reshape(predicted_pr, (len(predicted_pr))), pr)
    return_r_score = r2_score(np.reshape(predicted_data, (len(predicted_data))), test_y)
    p_mse = mean_squared_error(np.reshape(predicted_pr, (len(predicted_pr))), pr)
    return_mse = mean_squared_error(np.reshape(predicted_data, (len(predicted_data))), test_y)

    print(f"Regularizer for 1: {regularizer1} \nRegularizer for 2: {regularizer2} \nEpochs: {epochs}")
    print(f"Predicted life  value: {p_r_score} \nPredicted return  value: {return_r_score}"
          f"\nPredict life MSE: {p_mse} \nPredicted Return MSE: {return_mse}")


    MAE = eval.calcMAE(pr, predicted_pr)
    RMSE = eval.calcRMSE(pr, predicted_pr)
    MAPE = eval.calcMAPE(pr, predicted_pr )
    SMAPE = eval.calcSMAPE(pr, predicted_pr)
    print('Test MAE: %.8f' % MAE)
    print('Test RMSE: %.8f' % RMSE)
    print('Test MAPE: %.8f' % MAPE)
    print('Test SMAPE: %.8f' % SMAPE)



    dataset = []
    values = np.array([regularizer1, regularizer2, epochs, p_r_score, return_r_score, p_mse, return_mse])
    dataset.append(values)
    dataset = pd.DataFrame(dataset, columns=["regularizer1", "regularizer2", "epochs", "price_r_score", "return_r_score", "price_mse", "return_mse"])
    print(dataset)

    accuracy = []
    for i in range(len(pr)-1):
        acc = 100 - (np.abs(predicted_pr[i] - pr[i]))/pr[i] * 100
        accuracy.append(acc)
    average = np.mean(accuracy)
    std = np.std(accuracy)




    plt.figure(figsize=(16,8),dpi=80)
    plt.title("Prediction vs Actual")

    # plt.subplot(2, 1, 1)
    # plt.plot(np.arange(len(predicted_data)), np.reshape(test_y, (len(test_y))),
    #          np.reshape(predicted_data, (len(predicted_data))))
    # plt.title("Prediction vs Actual")
    # plt.ylabel("Log Return")

    # plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(predicted_pr)), np.reshape(pr, (len(pr))),
             np.reshape(predicted_pr, (len(predicted_pr))))
    plt.xlabel("Time stamp")
    plt.ylabel(" life")
    plt.show()




    return dataset, average, std


if __name__ == "__main__":
    dataset, average, std = nnmodel(20, 0.05, 0.01)
    print(f"life Accuracy Average = {average} \n life Accuracy Standard Deviation = {std}")

