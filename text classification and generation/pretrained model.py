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

###############################################################################
# import the pretrained embedding layer
import gensim

wv = gensim.models.KeyedVectors.load_word2vec_format("C:\\Users\\zhong\\Desktop\\work\\chinese embedding\\300d\\merge_sgns_bigram_char300.txt.bz2",
                                                     binary=False, encoding="utf8",  unicode_errors='ignore')  # C text format


pretrained_weights = wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape


max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)



def word2idx(word):
  return wv.vocab[word].index
def idx2word(idx):
  return wv.index2word[idx]

token_to_ix = { ch:i for i,ch in enumerate(wv.index2word) }
ix_to_token = { i:ch for i,ch in enumerate(wv.index2word) }

# Calculate word frequency
word_freq = {}
for word in all_tokens:
    word_freq[word] = word_freq.get(word, 0) + 1

ignored_words = set()
for k in set(all_tokens):
    if k not in wv.vocab:
        ignored_words.add(k)

# cut the text in semi-redundant sequences of SEQUENCE_LEN words
seq_length = 25 # number of steps to unroll the RNN for
STEP = 1
sentences = []
ignored = 0
for i in range(0, len(all_tokens) - seq_length, STEP):
    # Only add sequences where no word is in ignored_words
    if len(set(all_tokens[i: i+seq_length+1]).intersection(ignored_words)) == 0:
        sentences.append(all_tokens[i: i + seq_length])
    else:
        ignored = ignored+1
print('Ignored sequences:', ignored)
print('Remaining sequences:', len(sentences))

train_x = np.zeros([len(sentences), seq_length], dtype=np.int32)
train_y = np.zeros([len(sentences)], dtype=np.int32)
for i, sentence in enumerate(sentences):
  for t, word in enumerate(sentence[:-1]):
    train_x[i, t] = word2idx(word)
  train_y[i] = word2idx(sentence[-1])


from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
model.add(LSTM(units=emdedding_size))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

def sample(preds, temperature=1.0):
  if temperature <= 0:
    return np.argmax(preds)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def generate_next(text, num_generated=10):
  word_idxs = [word2idx(word) for word in text.lower().split()]
  for i in range(num_generated):
    prediction = model.predict(x=np.array(word_idxs))
    idx = sample(prediction[-1], temperature=0.7)
    word_idxs.append(idx)
  return ' '.join(idx2word(idx) for idx in word_idxs)

def on_epoch_end(epoch, _):
  print('\nGenerating text after epoch: %d' % epoch)
  texts = [
    '王',
    '韩',
    '钟'
  ]
  for text in texts:
    samp = generate_next(text)
    print('%s... -> %s' % (text, samp))

model.fit(train_x, train_y,
          batch_size=128,
          epochs=20,
          callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])









