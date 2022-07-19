import random

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

bostonDataset = load_boston()
features = pd.DataFrame(bostonDataset['data'])
features.columns = bostonDataset['feature_names']
label = pd.DataFrame(bostonDataset['target'])
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
