import random

import numpy as np
import pandas as pd

# Fetch dataset from original source since sklearn.datasets.load_boston is removed
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data_val = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

features = pd.DataFrame(data_val, columns=feature_names)
label = pd.DataFrame(target, columns=["target"])
data = label.join(features)

num = len(data)
index = list(range(num))
random.shuffle(index)
tt = data.iloc[index]
thre = num / 11
data_g = tt[:int(np.floor(thre * 5))]
data_h = tt[int(np.floor(thre * 5)):int(np.floor(thre * 10))]
data_t = tt[int(np.floor(thre * 10)):]

data_g.to_csv("./dataset/horizontal_house_price/house_price_1.csv")
data_h.to_csv("./dataset/horizontal_house_price/house_price_2.csv")
data_t.to_csv("./dataset/horizontal_house_price/house_price_test.csv")
