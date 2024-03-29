import numpy as np
import pandas as pd
import tensorflow as tf

# print(tf.__version__)
# need to import dataset
dataset = pd.read_csv(
    "Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Python/Churn_Modelling.csv"
)
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
# print(x)
# print(y)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])
# print(x)
# print(y)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough"
)
X = np.array(ct.fit_transform(x))
# print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

# Part 2 - Building the ANN

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Part 3 - Training the ANN

ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Part 4 - Making the predictions and evaluating the model

# print(
#     ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5
# )

y_pred = ann.predict(X_test)
y_pred = y_pred > 0.5
# print(
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
# )

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

# Part 5 - Evaluating, Improving and Tuning the ANN
