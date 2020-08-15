###############################################################################
from __future__ import absolute_import, division, print_function, unicode_literals

import os
# set working dir
os.chdir("C:\\Users\\zhong\\Dropbox\\idea\\MCMC project\\")

import tensorflow as tf
###############################################################################
# load the data folder
import pandas as pd

import pathlib
data_root = pathlib.Path(os.path.join("C:\\Users\\zhong\\Dropbox\\idea\\MCMC project\\","stories"))
all_story_paths = list(data_root.glob('*.txt'))

import numpy as np
import ntpath
story_names = [ntpath.splitext(ntpath.split(path)[1])[0] for path in all_story_paths]

fantacy = open(all_story_paths[0] , encoding="gbk")
fiction = open(all_story_paths[1] , encoding="gbk")

import re
def get_dat(dat):
    dat = re.split("\n",dat)
    dat = [re.sub("\s","",item) for item in dat]
    dat = np.delete(dat,np.where([item == "" for item in dat]))
    return dat

dat1 = get_dat(fantacy)
dat2 = get_dat(fiction)

label_names = np.array(["仙侠","科幻"])
label_to_index = dict((name, index) for index, name in enumerate(label_names))
y_label = [label_to_index[label] for label in y_label]


X = pd.concat(dat1,dat2)
y_label = pd.DataFrame(y_label)[np.array(X[[0]] != "")]
y_label = np.asarray(y_label)

X = np.asarray(X[np.array(X[[0]] != "")])
X = X.astype("str")

#one-hot encode target column
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_label)[:len(X)*0.75]
y_validate = to_categorical(y_label)[len(X)*0.75:len(X)*0.8]
y_test = to_categorical(y_label)[len(X)*0.8:]

# load model
###############################################################################
# import the embedding layer
import gensim

wv = gensim.models.KeyedVectors.load_word2vec_format("C:\\Users\\zhong\\Desktop\\work\\chinese embedding\\sgns.wiki.bigram-char",
                                                     binary=False, encoding="utf8",  unicode_errors='ignore')  # C text format

model = tf.keras.Sequential()
Embedding_layer = tf.keras.layers.Embedding(len(wv.vocab), wv.vector_size, 
                                            trainable=False)
Embedding_layer.weight = wv.vectors
model.add(Embedding_layer)
model.add(tf.keras.layers.Dense(wv.vector_size, activation='relu'))
model.add(tf.keras.layers.Dense(5,activation='softmax'))


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.summary()



import jieba

def preprocess(X):
    a = list(jieba.cut(str(X)))
    haha = list(filter(lambda x: x in wv.vocab, a))
    return haha


value = [preprocess(item) for item in X]

X1 = np.asarray(value)

len(X1[0])

for i in range(len(X1)):
    print(len(X1[i]))



X1.shape




