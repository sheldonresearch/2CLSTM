#!/usr/bin/env python
# encoding: utf-8

"""
@version: python2.7
@author: Xiangguo Sun
@contact: sunxiangguo@seu.edu.cn
@site: http://blog.csdn.net/github_36326955
@software: PyCharm
@file: 2CLSTM.py
@time: 17-7-27 5:15pm
"""
import os
import sys

import numpy as np
from keras.models import  Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense,Reshape, Flatten,LSTM, MaxPooling2D,Conv2D,Bidirectional, Concatenate,Dropout,Lambda, Input, TimeDistributed,MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from mytool import log_results,get_result,load_data_labels

"""
all you need to change is here:
"""
class_type = "OPN"



"""
you donnot need to change the following  
unless you want to change your model so that the training process can fit data well
"""

TRAIN_DATA_DIR ="../data/train_data/"+class_type+"/"
TEST_DATA_DIR ="../data/test_data/"+class_type+"/"



GLOVE_DIR = "/home/sunxiangguo/PycharmProjects/glove.6B/"
input_length = 500
MAX_NB_WORDS = 20000
w2vDimension = 100
VALIDATION_SPLIT = 0.2


embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]    # 单词的字符表示
    coefs = np.asarray(values[1:], dtype='float32')  # 单词的向量表示
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')




texts, labels, labels_index = load_data_labels(TRAIN_DATA_DIR)
texts_test, labels_test, labels_indeX_test = load_data_labels(TEST_DATA_DIR)
print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor

tokenizer = Tokenizer(MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
print("TEXT:",texts[0])
print("TEXT: ",sequences[0])
# tokenizer.fit_on_texts(texts_test)
sequences_test = tokenizer.texts_to_sequences(texts_test)




word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=input_length)

data_test = pad_sequences(sequences_test, maxlen=input_length)

labels_cat = to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels_cat.shape)



X_train = data
y_train_cat = labels_cat

y_train = np.asarray(y_train_cat.argmax(axis=1))

X_test = data_test
y_test = np.asarray(labels_test)
y_test_cat = to_categorical(y_test)

classes = len(labels_index)
print('Preparing embedding matrix.')

n_symbols = min(MAX_NB_WORDS, len(word_index))+1
embedding_weights = np.zeros((n_symbols, w2vDimension))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_weights[i] = embedding_vector


print("xtrain shape is：",X_train.shape)





hidden_dim_1 = 300
hidden_dim_2 = 300


callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0)]



#start our model
input_a = Input(shape=(input_length,))
embedding_layer=Embedding(output_dim=w2vDimension,
                        input_dim=n_symbols,
                        mask_zero=False,
                        weights=[embedding_weights],
                        input_length=input_length,
                        trainable=False)(input_a)

blstm = Bidirectional(LSTM(hidden_dim_1, return_sequences=True))(embedding_layer)

concatenate = Concatenate(axis=2)([blstm,embedding_layer])

concatenate=Dropout(rate=0.25)(concatenate)
# dense=Dense(hidden_dim_2,activation="relu")(concatenate)
# dense=Dropout(rate=0.25)(dense)
out = TimeDistributed(Dense(hidden_dim_2, activation="relu"))(concatenate)


reshape=Reshape(target_shape=(input_length,hidden_dim_2,1))(out)
# pool_rnn = MaxPooling2D(pool_size=(input_length,1))(dropout)  #8.9 modified
#flat=Flatten()(pool_rnn)



pool_rnn = MaxPooling2D(pool_size=(20,1))(reshape)  #8.9 modified
# #Lambda(lambda x: backend.max(x, axis = 1), output_shape = (hidden_dim_2, ))(dropout) # See equation (5).


conv1=Conv2D(filters=10,kernel_size=(1,1),activation="relu")(pool_rnn)
#conv1=Dropout(rate=0.25)(conv1)
conv2=Conv2D(filters=10,kernel_size=(2,1),activation="relu")(pool_rnn)
#conv2=Dropout(rate=0.25)(conv2)
conv3=Conv2D(filters=10,kernel_size=(3,1),activation="relu")(pool_rnn)
#conv3=Dropout(rate=0.25)(conv3)

max1 = MaxPooling2D(pool_size=(25, 1))(conv1)
max2 = MaxPooling2D(pool_size=(24, 1))(conv2)
max3 = MaxPooling2D(pool_size=(23, 1))(conv3)



concatenate = Concatenate(axis=1)([max1, max2, max3])
concatenate=Dropout(rate=0.2)(concatenate)

reshape_layer =Reshape(target_shape=(30,300,1))(concatenate)
max=MaxPooling2D(pool_size=(30,1))(reshape_layer)
flat=Flatten()(max)
dense=Dropout(rate=0.1)(flat)
dense=Dense(200,activation="relu")(dense)#(dense)


output = Dense(2,activation="softmax")(dense)

rcnn_model = Model(input=input_a, output=output)

from keras import metrics
rcnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
print(rcnn_model.summary())




bs = 32
ne = 100

hist = rcnn_model.fit(data, labels_cat,batch_size=bs,epochs=ne,
                      verbose=2,validation_split=0.25,callbacks=callbacks)

print("train process done!!")


rcnn_model.save("../model_para/RCNN3_c"+class_type+".h5")

y_proba = rcnn_model.predict(X_test, verbose=0)
y_pred = y_proba.argmax(axis=1)



path="../model_para/RCNN3_c" + class_type + "_train_test_details.txt"

accuracy,micro_precision,weighted_precision,macro_precision=get_result(y_test,y_pred)

log_results(path,accuracy,micro_precision,weighted_precision,macro_precision)
