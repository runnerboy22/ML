import numpy as np
import pandas as pd
import tensorflow as tf

#  import training set google stock price train
dataset_train = pd.read_csv(
    "templates/Part 3 - Recurrent Neural Networks 2/Google_Stock_Price_Train.csv"
)
training_set = dataset_train.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
