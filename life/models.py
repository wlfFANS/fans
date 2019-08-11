import keras.layers as kl
from keras.models import Model
from keras import regularizers
import pandas as pd
import numpy as np
import eval
import matplotlib.pyplot as plt
from bokeh.plotting import output_file, figure, show


class NeuralNetwork:
    def __init__(self, input_shape, s_or_return):
        self.input_shape = input_shape
        self.s_or_return = s_or_return

    def make_train_model(self):
        input_data = kl.Input(shape=(1, self.input_shape))
        lstm = kl.LSTM(10, input_shape=(1, self.input_shape), return_sequences=True, activity_regularizer=regularizers.l2(0.003),
                       recurrent_regularizer=regularizers.l2(0), dropout=0.5, recurrent_dropout=0.5)(input_data)
        perc = kl.Dense(10, activation="relu", activity_regularizer=regularizers.l2(0.005))(lstm)
        lstm2 = kl.LSTM(5, activity_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001),
                        dropout=0.5, recurrent_dropout=0.5)(perc)
        out = kl.Dense(1, activation="relu", activity_regularizer=regularizers.l2(0.001))(lstm2)

        model = Model(input_data, out)
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])

        # load data

        train = np.reshape(np.array(pd.read_csv("feat/autoencoded_train_data.csv", index_col=0)),
                           (len(np.array(pd.read_csv("feat/autoencoded_train_data.csv"))), 1, self.input_shape))
        train_y = np.array(pd.read_csv("feat/autoencoded_train_y.csv", index_col=0))
        # train_stock = np.array(pd.read_csv("train_stock.csv"))

        # train model

        model.fit(train, train_y, epochs=50)

        model.save("models/model.h5", overwrite=True, include_optimizer=True)

        test_x = np.reshape(np.array(pd.read_csv("feat/autoencoded_test_data.csv", index_col=0)),
                            (len(np.array(pd.read_csv("feat/autoencoded_test_data.csv"))), 1, self.input_shape))
        test_y = np.array(pd.read_csv("feat/autoencoded_test_y.csv", index_col=0))
        # test_stock = np.array(pd.read_csv("test_stock.csv"))

        stock_data_test = np.array(pd.read_csv('get_data/new_data_test.csv', index_col=0))

        print(model.evaluate(test_x, test_y))
        prediction_data = []
        stock_data = []
        for i in range(len(test_y)):
            prediction = (model.predict(np.reshape(test_x[i], (1, 1, self.input_shape))))
            prediction_data.append(np.reshape(prediction, (1,)))
            # prediction_corrected = (prediction_data - np.mean(prediction_data))/np.std(prediction_data)
            stock_price = (np.exp(np.reshape(prediction, (1,)))*(stock_data_test[i]))
            stock_price = pd.DataFrame(stock_price).fillna(3).values
            stock_data.append(stock_price[0])
            stock_data[:] = [i - (float(stock_data[0])-float(stock_data_test[0])) for i in stock_data]





        # plt.plot(stock_data)
        # plt.plot(stock_data_test )
        # stock = pd.DataFrame(stock_data, index=None)
        # stock.to_csv("samples/predicted.csv")
        # stock_test = pd.DataFrame(stock_data_test, index=None)
        # stock_test.to_csv("samples/actual.csv")
        # print(stock_data)
        # plt.show()



        accuracy = []
        for i in range(len(stock_data_test)-1):
            acc = float(100 - ((np.abs(stock_data[i] - (stock_data_test[i]))) / (stock_data_test[i]) * 100))
            if (1 < acc < 100):
                accuracy.append(acc)
            # print(acc)
        # print(type(accuracy))
        # print(accuracy)
        # print(np.array(accuracy))
        average = np.sum(accuracy) / len(accuracy)
        std = np.std(accuracy)
        print("accuracy average", average)
        print("accuracy std", std)







        if self.s_or_return:


            plt.figure(figsize=(18, 8), dpi=150)
            # plt.subplot(1, 2, 1)
            plt.plot(stock_data ,linestyle='-',label ='predicted' )
            # plt.title('predicted' )
            # plt.subplot(1, 2, 2)
            plt.plot(stock_data_test,linestyle=':',label ='actual')
            # plt.title('actual')
            # plt.subplot(1, 3, 3)
            # plt.plot(stock_data )
            # plt.plot(stock_data_test)
            # plt.title('predicted vs actual')

            # stock = pd.DataFrame(stock_data, index=None)
            # stock.to_csv("samples/predicted.csv")
            # stock_test = pd.DataFrame(stock_data_test, index=None)
            # stock_test.to_csv("samples/actual.csv")
            # print(stock_data)

            plt.legend(loc='upper right')
            plt.show()
        # else:
        #     plt.plot(prediction_data)
        #     plt.plot(test_y)
        #     plt.show()

        stock_data_test =pd.DataFrame(stock_data_test).fillna(0).iloc[0:121].values
        stock_data = pd.DataFrame(stock_data).fillna(0).iloc[0:121].values
        MAE = eval.calcMAE(stock_data_test, stock_data)
        RMSE = eval.calcRMSE(stock_data_test, stock_data)
        MAPE = eval.calcMAPE(stock_data_test, stock_data)
        SMAPE = eval.calcSMAPE(stock_data_test, stock_data)

        print('Test MAE: %.8f' % MAE)
        print('Test RMSE: %.8f' % RMSE)
        print('Test MAPE: %.8f' % MAPE)
        print('Test SMAPE: %.8f' % SMAPE)


if __name__ == "__main__":
    model = NeuralNetwork(20, True)
    model.make_train_model()
