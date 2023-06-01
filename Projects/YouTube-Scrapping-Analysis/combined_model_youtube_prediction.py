# -*- coding: utf-8 -*-
"""Combined Model- YouTube Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ceW1KmnzjbMLXUI4QDcs9d2r4YraQVNQ
"""

import pandas as pd
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

import cv2
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input,Dense, Dropout, Flatten, LSTM, Embedding,concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D

import joblib

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('words')

emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)

regular_punct = list(string.punctuation)
stop_words = set(stopwords.words('english'))
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
english_words = set(words.words())

def clean_text(text):

  #remove emoji
  sentence = emoji_pattern.sub(r'', text)

  #removing non english words and converting to lower
  sentence = ' '.join([word.lower() for word in sentence.split() if word not in (english_words)])

  #remove stop words
  sentence =  ' '.join([word for word in sentence.split() if word not in (stop_words)])

  # remove punctuation
  for punc in regular_punct:
        if punc in sentence:
            sentence = sentence.replace(punc, ' ')
  
  #lemmatize the words
  lemmatized_sentence = ""
  for w in w_tokenizer.tokenize(sentence):
        lemmatized_sentence = lemmatized_sentence + lemmatizer.lemmatize(w) + " "
  
  return lemmatized_sentence.strip()

def bin_subs(text):

  subs = int(text)/1000000
  bin_subs = 0

  if(subs < 1):
    bin_subs = 0
  elif (1 < subs < 5):
    bin_subs = 1
  elif (5 < subs < 10):
    bin_subs = 2
  elif (10 < subs < 15):
    bin_subs = 3
  elif (15 < subs < 20):
    bin_subs = 4
  else:
    bin_subs = 5

  return bin_subs

data = pd.read_csv("YoutubeData_new.csv")
data['clean_text']=data['Title'].apply(lambda x : clean_text(x))
data['bin_subs'] = data['Subscribers'].apply(lambda x: bin_subs(x))

# tokenize sentences
tokenizer = Tokenizer(1000)
tokenizer.fit_on_texts(data['clean_text'])
word_index = tokenizer.word_index
# convert train dataset to sequence and pad sequences
clean_text = tokenizer.texts_to_sequences(data['clean_text']) 
clean_text = pad_sequences(clean_text, padding='pre', truncating= 'pre', maxlen=10) / 1000

arr_images =  joblib.load(f'drive//MyDrive//Image_array.joblib')

def linear_model(input_layer, input_shape):

  x = Dense(16, input_dim = input_shape, activation = 'relu')(input_layer)  
  x = Dense(2, activation = 'relu')(x)

  return x

def title_model(input_layer, input_shape):

  x = Embedding(1000, input_shape, input_length=input_shape)(input_layer)
  x = LSTM(128, return_sequences=True, input_shape= (input_shape, 1))(x)
  x = LSTM(64, return_sequences=False)(x)
  x = Flatten()(x)
  x = Dense(64, activation = 'relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(16, activation = 'relu')(x)

  return x

def thumbnail_model(input_layer, input_shape):

  x = Conv2D(64, kernel_size = (2,2), strides=2, padding="same", activation = 'relu', input_shape = input_shape)(input_layer)
  x = Conv2D(64, kernel_size = (2,2), strides=2, padding="same", activation  = 'relu')(x)
  x = MaxPooling2D(pool_size = (2,2), padding="same")(x)
  x = Dropout(0.2)(x)
  x = Conv2D(32, kernel_size = (2,2), strides=2, padding="same", activation = 'relu')(x)
  x = MaxPooling2D(pool_size = (2,2), padding="same")(x)
  x = Conv2D(32, kernel_size = (2,2), strides=2, padding="same", activation = 'relu')(x)
  x = MaxPooling2D(pool_size = (2,2), padding="same")(x)
  x = Dropout(0.2)(x)
  x = Flatten()(x)
  x = Dense(1024, activation = 'relu')(x)

  return x

numerical_data = data[['LikesCount', 'CommentCount', 'bin_subs']]

Y = data['ViewsCount']

train_len = int(data.shape[0] * 0.8)
xtrain_txt = clean_text[:train_len]
xtest_txt = clean_text[train_len:]
xtrain_img = arr_images[:train_len]
xtest_img = arr_images[train_len:]
xtrain_num = numerical_data[:train_len]
xtest_num = numerical_data[train_len:]
ytrain = Y[:train_len]
ytest = Y[train_len:]

num_input = Input(shape=3)
text_input = Input(shape=10)
image_input = Input(shape=(90,90,3))

num_layers = linear_model(num_input, 3)
text_layers = title_model(text_input, 10)
image_layers = thumbnail_model(image_input, (90,90,3))

out=concatenate([num_layers,text_layers,image_layers], axis=-1)
output=Dense(1, activation='relu')(out)
model = Model(inputs=[num_input,text_input, image_input], outputs=output)

loss = tf.keras.losses.MeanSquaredError()
metric = [tf.keras.metrics.RootMeanSquaredError()]
optimizer = tf.keras.optimizers.Adam()
early_stopping = [tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 5)]

model.compile(loss = loss, metrics = metric, optimizer = optimizer)

history = model.fit([xtrain_num,xtrain_txt, xtrain_img], ytrain, epochs = 300, callbacks = early_stopping)

ypred = model.predict([xtest_num[:2225], xtest_txt[:2225],xtest_img])
mse = mean_squared_error(ytest[:2225], ypred[:2225])
print(f'Mean Squarred Error: {mse}')
print(f'Root Mean Squarred Error: {np.sqrt(mse)}')
print(f'R2 score: {r2_score(ytest[:2225], ypred[:2225])}')

print(ytest.iloc[100])
print(ypred[100][0])

model.save('combined_model2.h5')

