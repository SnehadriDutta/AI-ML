import numpy as np
import pandas as pd
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Flatten, Bidirectional, SimpleRNN
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xg
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
tokenizer = Tokenizer(1000)
stop_words = set(stopwords.words('english'))

def remove_punctuation(text):
    regular_punct = list(string.punctuation)
    for punc in regular_punct:
        if punc in text:
            text = text.replace(punc, '')
    return text.lower().strip()


def lemmatize_text(text):
    st = ""
    for w in w_tokenizer.tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st

def clean_Text_col(df):

    data = df.copy()

    data['clean_text'] = data['Title'].apply(lambda x: remove_punctuation(x))
    data['clean_text'] = data['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    data['clean_text'] = data['clean_text'].apply(lambda x: lemmatize_text(x))


    tokenizer.fit_on_texts(data['clean_text'])
    word_index = tokenizer.word_index
    # convert train dataset to sequence and pad sequences
    clean_text = tokenizer.texts_to_sequences(data['clean_text'])
    clean_text = pad_sequences(clean_text, padding='pre', maxlen=20)

    return (clean_text, data['ViewsCount'])

def create_title_model(data):

    (X, Y) = clean_Text_col(data)
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler(feature_range=(0, 1))
    ytrain = scaler.fit_transform(np.array(ytrain).reshape(-1, 1))
    ytest = scaler.transform(np.array(ytest).reshape(-1, 1))

    loss = tf.keras.losses.MeanSquaredError()
    metric = [tf.keras.metrics.RootMeanSquaredError()]
    optimizer = tf.keras.optimizers.Adam()
    early_stopping = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)]

    model = Sequential()
    model.add(Embedding(1000, xtrain.shape[1], input_length=xtrain.shape[1]))
    model.add(LSTM(256, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128, return_sequences=False))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(96, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss=loss, metrics=metric, optimizer=optimizer)

    model.fit(xtrain, ytrain, epochs=300, callbacks=early_stopping)

    return model

def process_text(text):

    text = remove_punctuation(text)
    text = ' '.join([word for word in text.split() if word not in (stop_words)])
    clean_Text = lemmatize_text(text)
    clean_text = tokenizer.texts_to_sequences(data['clean_text'])
    clean_text = pad_sequences(clean_text, padding='pre', maxlen=20)

    return clean_Text