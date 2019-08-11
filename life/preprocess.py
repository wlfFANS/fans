import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt


class PreProcessing:
    def __init__(self, split, feature_split):
        self.split = split
        self.feature_split = feature_split
        self.s_data = pd.read_csv("get_data/b.csv")

    # wavelet transform and create autoencoder data
    def make_wavelet_train(self):
        train_data = []
        test_data = []
        log_train_data = []
        for i in range((len(self.s_data)//10)*10 - 11):
            # print(i)
            train = []
            log_ret = []
            for j in range(1, 5):
                x = np.array(self.s_data.iloc[i: i + 11, j])
                (ca, cd) = pywt.dwt(x, "haar")
                cat = pywt.threshold(ca, np.std(ca), mode="soft")
                cdt = pywt.threshold(cd, np.std(cd), mode="soft")
                tx = pywt.idwt(cat, cdt, "haar")
                log = np.diff(np.log(tx))*100
                macd = np.mean(x[5:]) - np.mean(x)
                # ma = np.mean(x)
                sd = np.std(x)
                log_ret = np.append(log_ret, log)
                x_tech = np.append(macd*10, sd)
                train = np.append(train, x_tech)
            train_data.append(train)
            log_train_data.append(log_ret)
        trained = pd.DataFrame(train_data)
        trained.to_csv("preprocess/indicators.csv")
        log_train = pd.DataFrame(log_train_data, index=None)
        log_train.to_csv("preprocess/log_train.csv")

        rbm_train = pd.DataFrame(log_train_data[0:int(self.split*self.feature_split*len(log_train_data))], index=None)
        rbm_train.to_csv("preprocess/rbm_train.csv")
        rbm_test = pd.DataFrame(log_train_data[int(self.split*self.feature_split*len(log_train_data))+1:
                                               int(self.feature_split*len(log_train_data))])
        rbm_test.to_csv("preprocess/rbm_test.csv")
        for i in range((len(self.s_data) // 10) * 10 - 11):
            y = 100*np.log(self.s_data.iloc[i + 11, 4] / self.s_data.iloc[i + 10,4])
            test_data.append(y)
        test = pd.DataFrame(test_data)
        test.to_csv("preprocess/test_data.csv")

    def make_test_data(self):
        test_s = []

        for i in range((len(self.s_data) // 10) * 10 -11):
            l = self.s_data.iloc[i+11, 4]
            test_s.append(l)
            test = pd.DataFrame(test_s)
            test.to_csv("preprocess/test_stock.csv")

        s_test_data = np.array(test_s)[int(self.feature_split*len(test_s) +
                                               self.split*(1-self.feature_split)*len(test_s)):]
        st = pd.DataFrame(s_test_data, index=None)
        st.to_csv("get_data/new_data_test.csv")



if __name__ == "__main__":
    preprocess = PreProcessing(0.7, 0.3)
    preprocess.make_wavelet_train()
    preprocess.make_test_data()

