###############################################################################
# load data
f = open("C:\\Users\\zhong\\Dropbox\\idea\\MCMC project\\stories\\fanrenxiuxianchuan_wangyu.txt" , encoding="gbk")
data = f.read()

###############################################################################
# words tokens
import jieba
import numpy as np
import re
dat = re.split("\n",data)
dat = [re.sub("\s","",item) for item in dat]
dat = [re.sub("-","",item) for item in dat]
dat = [re.sub("☆","",item) for item in dat]
dat = np.delete(dat,np.where([item == "" for item in dat]))
dat2 = ''.join(dat)
all_tokens = list(jieba.cut(str(dat2)))

# Calculate word frequency
word_freq = {}
for word in all_tokens:
    word_freq[word] = word_freq.get(word, 0) + 1

MIN_WORD_FREQUENCY = 5

ignored_words = set()
for k, v in word_freq.items():
    if word_freq[k] < MIN_WORD_FREQUENCY:
        ignored_words.add(k)

words = set(all_tokens)
print('Unique words before ignoring:', len(words))
print('Ignoring words with frequency <', MIN_WORD_FREQUENCY)
words = sorted(set(words) - ignored_words)
print('Unique words after ignoring:', len(words))

VOCAB_SIZE = len(words)

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

# cut the text in semi-redundant sequences of SEQUENCE_LEN words
seq_length = 25 # number of steps to unroll the RNN for
STEP = 1
sentences = []
next_words = []
ignored = 0
for i in range(0, len(all_tokens) - seq_length, STEP):
    # Only add sequences where no word is in ignored_words
    if len(set(all_tokens[i: i+seq_length+1]).intersection(ignored_words)) == 0:
        sentences.append(all_tokens[i: i + seq_length])
        next_words.append(all_tokens[i + seq_length])
    else:
        ignored = ignored+1
print('Ignored sequences:', ignored)
print('Remaining sequences:', len(sentences))

from sklearn.model_selection import train_test_split
sentences, next_words, sentences_test, next_words_test = train_test_split(sentences, next_words, test_size=0.33)

import tensorflow as tf

dropout = 0.1

model = tf.keras.Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128), input_shape=(seq_length, len(words))))
if dropout > 0:
    model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(len(words)))
model.add(tf.keras.layers.Activation('softmax'))

def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, seq_length, len(words)), dtype=np.bool)
        y = np.zeros((batch_size, len(words)), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index]):
                x[i, t, word_indices[w]] = 1
            y[i, [word_indices[item] for item in next_word_list[index]]] = 1
            index = index + 1
            if index == len(sentence_list):
                index = 0
        yield x, y


file_path = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % (
    len(words),
    seq_length,
    MIN_WORD_FREQUENCY
)
checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True)
print_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch,logs : None)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5)
callbacks_list = [checkpoint, print_callback, early_stopping]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

BATCH_SIZE= 25

model.fit_generator(generator(sentences, next_words, BATCH_SIZE),
    steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1,
    epochs=1,
    callbacks=callbacks_list,
    validation_data=generator(sentences_test, next_words_test, BATCH_SIZE), 
             validation_steps=int(len(sentences_test)/BATCH_SIZE) + 1)


def generate_text(model, length):
    ix = [np.random.randint(VOCAB_SIZE)]
    y_char = [indices_word[ix[-1]]]
    X = np.zeros((1, length, VOCAB_SIZE))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(indices_word[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(indices_word[ix[-1]])
    return ('').join(y_char)


generate_text(model, 50)