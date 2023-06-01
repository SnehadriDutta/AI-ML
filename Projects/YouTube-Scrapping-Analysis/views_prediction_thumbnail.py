# -*- coding: utf-8 -*-
"""Youtube  Thumbnail Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1biNF6QCpKOMZJ8mpeNogJw3V5yttIHa3
"""

import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D

data = pd.read_csv("YoutubeData.csv")
data.head()

X = []
Y = []

for i in range(data.shape[0]):
  id = data.iloc[i]['Id']
  viewcount = data.iloc[i]['ViewsCount']
  img_path = f'drive//MyDrive//Thumbnails//{id}.jpg'

  image = cv2.imread(img_path) 
  image = cv2.resize(image, (90,90), interpolation= cv2.INTER_LINEAR)  
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
  # Append the image and its corresponding label to the output
  X.append(image)
  Y.append(viewcount)
  print(i)

X = np.array(X, dtype = 'float32') / 255.0
Y = np.array(Y, dtype = 'int32')

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2, random_state = 42)

scaler = MinMaxScaler(feature_range = (0,1))
ytrain = scaler.fit_transform(np.array(ytrain).reshape(-1,1))
ytest = scaler.fit_transform(np.array(ytest).reshape(-1,1))

loss = tf.keras.losses.MeanSquaredError()
metric = [tf.keras.metrics.RootMeanSquaredError()]
optimizer = tf.keras.optimizers.Adam()
early_stopping = [tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 10)]


model = Sequential()
model.add(Conv2D(64, kernel_size = (2,2), strides=2, padding="same", activation = 'relu', input_shape = (90,90,3)))
model.add(Conv2D(64, kernel_size = (2,2), strides=2, padding="same", activation  = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), padding="same"))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size = (2,2), strides=2, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), padding="same"))
model.add(Conv2D(32, kernel_size = (2,2), strides=2, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), padding="same"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

model.compile(loss = loss, metrics = metric, optimizer = optimizer)

r = model.fit(xtrain, ytrain, epochs = 300, batch_size=32, callbacks = early_stopping)

ypred = model.predict(xtest)
mse = mean_squared_error(ytest, ypred.flatten())
print(f'Mean Squarred Error: {mse}')
print(f'Root Mean Squarred Error: {np.sqrt(mse)}')
print(f'R2 score: {r2_score(ytest,ypred.flatten())}')

print(scaler.inverse_transform(ytest[122].reshape(-1,1)))
print(scaler.inverse_transform(ypred[122].reshape(-1,1)))

#model.save("thumbnail_model.h5")

#model.save("drive//MyDrive//thumnail_model.h5")