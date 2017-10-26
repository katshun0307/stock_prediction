import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import h5py
import sys, os

sys.path.append(os.pardir)
from sklearn.utils import shuffle


'''
データの取得
'''
# problem = np.load("apple_stocks.npy")
problem = np.load('nyse_stocks.npy')

length_of_sequences = len(problem) - 1
maxlen = 50

data = []
target = []

for i in range(0, length_of_sequences - maxlen + 1):
    data.append(problem[i: i + maxlen])
    target.append(problem[i + maxlen])

X = np.array(data).reshape(len(data), maxlen, 20)
Y = np.array(target).reshape(len(data), 20)

N_train = int(len(data) * 0.9)
N_validation = len(data) - N_train

X_train, X_validation, Y_train, Y_validation = \
    train_test_split(X, Y, test_size=N_validation)


'''
モデル設定
'''
n_in = len(X[0][0])  # 1
n_hidden = 30
n_out = len(Y[0])  # 1


def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)


try:
    model = keras.models.load_model('./stock_lstm.h5')
    print("loaded model")
except Exception as e:
    print(e)
    print("model did not exist.\nmaking model")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    model = Sequential()
    model.add(LSTM(n_hidden,
                   kernel_initializer=weight_variable,
                   input_shape=(maxlen, n_in)))
    model.add(Dense(n_out, kernel_initializer=weight_variable))
    model.add(Activation('linear'))

    try:
        model.load_weights('lstm_weights.h5')
        print("loaded weights")
    except Exception as e:
        print("could not load model")

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer)

'''
モデル学習
'''
epochs = 5
batch_size = 10

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_validation, Y_validation),
          callbacks=[early_stopping])


print("finished learning model...")

model.save_weights('lstm_weights.h5')

# print("saving model")
# model.save('stock_lstm.h5')
# print("save finished")


'''
出力を用いて予測
'''
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

original = np.array(original).transpose()
predicted = np.array(predicted).transpose()
problem = np.array(problem).transpose()

'''
グラフで可視化
'''

# write graph of first company
plt.rc('font', family='serif')
plt.figure()
plt.ylim([0, 200])
plt.plot(problem[:1], linestyle='dotted', color='#aaaaaa')
plt.plot(original[:1], linestyle='dashed', color='black')
plt.plot(predicted[:1], color='black')
plt.show()
