"""
    Example taken from https://towardsai.net/p/deep-learning/house-price-predictions-using-keras
    Author: towardsai
"""

# import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
# Creating a Neural Network Model
from tensorflow import keras
from tensorflow.keras import layers


Data = pd.read_csv('./input/kc_house_data.csv')

# let's drop unnecessory featurs
Data = Data.drop('id',axis=1)
Data = Data.drop('zipcode',axis=1)

Data = Data.drop('date',axis=1)

X = Data.drop('price',axis =1).values
y = Data['price'].values

#splitting Train and Test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

#standardization scaler - fit&transform on train, fit only on test
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float64))
X_test = s_scaler.transform(X_test.astype(np.float64))

print(X_train.shape)

# having 17 nueron is based on the number of available features

inputs = keras.Input(shape=(17,))
x = layers.Dense(17, activation="relu")(inputs)
x = layers.Dense(17, activation="relu")(x)
x = layers.Dense(17, activation="relu")(x)
x = layers.Dense(1, activation="relu")(x)
model = keras.Model(inputs, x)

model.compile(optimizer='adam',loss='mse')

# training
model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=128,epochs=400)
model.summary()

# evaluation
y_pred = model.predict(X_test)

# evaluation metrics
# explained variance score: best possible score is 1 and lower values are worse
print('Mean Absolute Error: {:.2f}'.format(metrics.mean_absolute_error(y_test, y_pred)))
print('Mean Squared Error: {:.2f}'.format(metrics.mean_squared_error(y_test, y_pred)))
print('Root Mean Squared Error: {:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
print('Variance score is: {:.2f}'.format(metrics.explained_variance_score(y_test,y_pred)))

# example
single_house = Data.drop('price',axis = 1).iloc[0]
single_house = s_scaler.transform(single_house.values.reshape(-1,17))
prediction = model.predict(single_house)

print(f"Real value: {Data['price'][0]} ==> {prediction[0][0]}")