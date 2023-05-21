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

def create_X_Y(data):

    X = []
    Y = []

    for i in range(data.shape[0]):
        id = data.iloc[i]['Id']
        viewcount = data.iloc[i]['ViewsCount']
        img_path = f'drive//MyDrive//Thumbnails//{id}.jpg'

        image = cv2.imread(img_path)
        image = cv2.resize(image, (90, 90), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Append the image and its corresponding label to the output
        X.append(image)
        Y.append(viewcount)
        print(i)

    X = np.array(X, dtype='float32') / 255.0
    Y = np.array(Y, dtype='int32')

    return (X,Y)

def create_thumbnail_model(data):

    X, Y = create_X_Y(data)

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler(feature_range=(0, 1))
    ytrain = scaler.fit_transform(np.array(ytrain).reshape(-1, 1))
    ytest = scaler.fit_transform(np.array(ytest).reshape(-1, 1))

    loss = tf.keras.losses.MeanSquaredError()
    metric = [tf.keras.metrics.RootMeanSquaredError()]
    optimizer = tf.keras.optimizers.Adam()
    early_stopping = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)]

    model = Sequential()
    model.add(Conv2D(512, kernel_size=(2, 2), activation='relu', input_shape=(90, 90, 3)))
    model.add(Conv2D(256, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss=loss, metrics=metric, optimizer=optimizer)

    model.fit(xtrain, ytrain, epochs=300, batch_size=128, callbacks=early_stopping)

    return model

def process_image(filename):

    image = cv2.imread(filename)
    image = cv2.resize(image, (90, 90), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return (image / 255.0)