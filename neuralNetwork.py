# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:41:56 2024

@author: ibsens

Train Neural Translation Model
"""

from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint


# load the clean data set : )
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

# load them data sets
dataset = load_clean_sentences('english-magyar-both.pkl')
train = load_clean_sentences('english-magyar-train.pkl')
test = load_clean_sentences('english-magyar-test.pkl')

# fit a tokenizer to map words to integers as needed for modeling
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocav Size: %d' % eng_vocab_size)
print('English Max Length : %d' % (eng_length))

# prepare magyar token
magyar_tokenizer = create_tokenizer(dataset[:,1])
magyar_voca_size = len(magyar_tokenizer.word_index) + 1
magyar_length = max_length(dataset[:, 1])
print('Magyar Vocabular Size: %d' % magyar_voca_size)
print('Magyar Max Length: %d' % (magyar_length))

# encode and pad sequences 
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad seq with 0 values
    X = pad_sequences(X, maxlen = length, padding = 'post')
    return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes = vocab_size)
        ylist.append(encoded)
        y = array(list)
        y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
        return y

# prepare training data
trainX = encode_sequences(magyar_tokenizer, magyar_length, train[:,1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:,0])
trainY = encode_output(trainY, eng_vocab_size)

#prepare validation data
testX = encode_sequences(magyar_tokenizer, magyar_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:,0])
testY = encode_output(testY, eng_vocab_size)


# define NMT model
def define_model(src_vocab, tar_vocab, src_timestamps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length = src_timestamps, mask_zero = True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences = True))
    model.add(TimeDistributed(Dense(tar_vocab, activation = 'softmax')))
    return model

# define model
model = define_model(magyar_voca_size, eng_vocab_size, magyar_length, eng_length, 256)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
# summarize defined model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes = True)

# training the model with 30 epochs and a batch size of 64 examples
# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
model.fit(trainX, trainY, epochs = 30, batch_size = 64, validation_data = (testX, testY), callbacks = [checkpoint], verbose = 2)