""" stock_prediction / stock_lstm_validation.py :  """

"""

"""
__author__ = "Shuntaro Katsuda"

# encoding: utf-8

import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import initializers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import h5py
import sys, os

sys.path.append(os.pardir)
from sklearn.utils import shuffle

""" ========== """
LEARNING_DATA = "validation_nyse_stocks.npy"
WEIGHTS_FILE = "lstm_weights_normalized_new.h5"
EPOCHS = 0  # do not change
MAXLEN = 200 # same as stock_lstm.py
BATCH_SIZE = 10 # no needed
""" ========== """



'''
get learning data
'''
# problem = np.load("apple_stocks.npy")
problem = np.load(LEARNING_DATA)

# normalize data
problem = preprocessing.scale(problem)
print("normalized!")

length_of_sequences = len(problem) - 1

maxlen = MAXLEN

data = []
target = []

for i in range(0, length_of_sequences - maxlen + 1):
    data.append(problem[i: i + maxlen])
    target.append(problem[i + maxlen])

X = np.array(data).reshape(len(data), maxlen, 20)
Y = np.array(target).reshape(len(data), 20)

N_train = int(len(data) * 0.9)
N_validation = len(data) - N_train

# X_train, X_validation, Y_train, Y_validation = \
#     train_test_split(X, Y, test_size=N_validation)

split_index = N_train
X_train = X[:split_index]
X_validation = X[split_index:]
Y_train = Y[:split_index]
Y_validation = Y[split_index:]


'''
build model
'''
n_in = len(X[0][0])  # 1
n_hidden = 30
n_out = len(Y[0])  # 1


def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)

setattr(initializers, 'weight_variable', weight_variable)

# define model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

model = Sequential()
model.add(LSTM(n_hidden,
               kernel_initializer=weight_variable,
               # kernel_initializer=lambda shape: np.random.normal(scale=.01, size=shape),
               input_shape=(maxlen, n_in)))
model.add(Dense(n_out, kernel_initializer=weight_variable))
# model.add(Dense(n_out, kernel_initializer=lambda shape: np.random.normal(scale=.01, size=shape)))
model.add(Activation('linear'))

# load weights if exists
try:
    model.load_weights(WEIGHTS_FILE)
    print("loaded weights")
except Exception as e:
    print("could not load model")


# compile model
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='mean_squared_error',
              optimizer=optimizer)

'''
optimize model
'''
epochs = EPOCHS
batch_size = BATCH_SIZE

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_validation, Y_validation),
          callbacks=[early_stopping])


print("finished learning model...")
model.save('lstm_model.h5')

print("saving weights...")
model.save_weights(WEIGHTS_FILE)
print("saved weights")


# print("saving model")
# model.save('stock_lstm.h5')
# print("save finished")


'''
predict
'''
print("predicting...")
truncate = maxlen
Z = X[:1]  # 元データの最初の一部だけ切り出し

original = [problem[i] for i in range(maxlen)]
predicted = [None for i in range(maxlen)]

for i in range(length_of_sequences - maxlen + 1):
    z_ = Z[-1:]
    y_ = model.predict(z_)
    sequence_ = np.concatenate(
        (z_.reshape(maxlen, n_in)[1:], y_),
        axis=0).reshape(1, maxlen, n_in)
    Z = np.append(Z, sequence_, axis=0)
    predicted.append(y_.reshape(-1))
print("finished predicting")

print("writing graph")

#original = np.array(original).transpose()
#predicted = np.array(predicted).transpose()
#problem = np.array(problem).transpose()

print(predicted)
'''
visualize with graph
'''

# write graph of first brand
# plt.rc('font', family='serif')
# x = np.arange(0, 700, 1)
# plt.figure()
# plt.ylim([0, 200])
# plt.plot(problem, linestyle='dotted', color='#aaaaaa')
# plt.plot(original, linestyle='dashed', color='black')
# plt.plot(x, predicted, color='black')
# plt.show()

# build new array for graph
#predicted_ = predicted[50:]
predicted_ = []
original_ = []
problem_ = []
for prices in predicted:
    try:
        predicted_.append(prices[0])
    except Exception as e:
        predicted_.append(None)

for prices in original:
    original_.append(prices[0])

for prices in problem:
    problem_.append(prices[0])

plt.plot(predicted_)
plt.plot(problem_)
plt.plot(original_)
plt.show()



