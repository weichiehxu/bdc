# 1 acc: 0.9867 - val_loss: 0.0477 - val_acc: 0.9838
# 2 acc: 0.9787 - val_loss: 0.0475 - val_acc: 0.9930
import numpy as np
import pandas as pd
import gensim
import multiprocessing
from preprocess import PatternTokenizer

#train = pd.read_csv('train.csv')
#test = pd.read_csv('test.csv')

# preprocess fucking data
# remove non-letter and make lowercase
#train['comment_text'] = train['comment_text'].str.replace('[^a-zA-Z]',' ').str.lower()
#test['comment_text'] = test['comment_text'].str.replace('[^a-zA-Z]',' ').str.lower()



# spilt fucking data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

tokenizer = PatternTokenizer()
train["comment_text"] = tokenizer.process_ds(train["comment_text"]).str.join(sep=" ")
test["comment_text"] = tokenizer.process_ds(test["comment_text"]).str.join(sep=" ")
train_split = train['comment_text'].str.split()
test_split = test['comment_text'].str.split()

train_list = train_split.values.tolist()
test_list = test_split.values.tolist()
# print(multiprocessing.cpu_count()) TMD I only 4 core to run

# use default w2c parameters
word2vec = gensim.models.word2vec.Word2Vec(sentences=train_list, workers=4)
from collections import defaultdict
vocab = defaultdict(int)
for k, v in word2vec.wv.vocab.items():
    vocab[k] = v.index

max([v for k,v in vocab.items()])

train_ind = [[vocab[w] for w in train_split[i]] for i in range(len(train_split))]
test_ind = [[vocab[w] for w in test_split[i]] for i in range(len(test_split))]

from keras.preprocessing.sequence import pad_sequences
train_padded = pad_sequences(train_ind,maxlen=100,truncating='pre')
test_padded = pad_sequences(test_ind,maxlen=100,truncating='pre')

emb_layer = word2vec.wv.get_keras_embedding()


from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model

embed_size = 50  # how big is each word vector
max_features = 20000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100  # max number of words in a comment to use

inp = Input(shape=(maxlen,))  # define input
x = emb_layer(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = BatchNormalization()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)  # define output
model = Model(inputs=inp, outputs=x)

import keras.backend as K


def loss(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred)


model.compile(loss=loss, optimizer='nadam', metrics=['accuracy'])
y = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
from keras import callbacks


def schedule(ind):
    a = [0.002, 0.003, 0.000]
    return a[ind]


lr = callbacks.LearningRateScheduler(schedule)
print("Training--------")
model.fit(x=train_padded, y=y, validation_split=.1, epochs=2, batch_size=32)

test_pred = model.predict(x=test_padded)
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission[list_classes] = test_pred
sample_submission.to_csv('word2vec+LSTM1.csv', index=False)