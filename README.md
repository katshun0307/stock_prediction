# Stock Prediction

This LSTM network collects stock prices of 20 companies in NYSE, and tries to predict these future prices.

## collect stock data
Collection of stocks are defined in `stock_price_collector.py`.

The stock prices are saved in `.npy` format. Two files are saved, each for training and validation of the network.

The time range for data are defined like this.

```python
training_start_time = datetime.datetime(2013, 1, 1)
training_end_time = datetime.datetime(2015, 1, 1)
validation_start_time = training_end_time
validation_end_time = datetime.datetime.today()
```
The data is then saved.

```python
np.save("training_nyse_stocks.npy", training_nyse_open)
np.save("validation_nyse_stocks.npy", validation_nyse_open)
```

## train neural network

The neural network is defined and then trained in `stock_lstm.py`.

The weights from the previous training are loaded.

```python
# load weights if exists
try:
    model.load_weights(WEIGHTS_FILE)
    print("loaded weights")
except Exception as e:
    print("could not load model")

```
Then, the network is trained, and then the weights are saved to  `lstm_weights_normalizesd_new.h5`.
You need to download `h5py` from pip and install `hdf5` to use weight saving.

+ On mac run
```
brew install hdf5
pip install h5py
```


## validate neural network

Validation of the network can be done in `validation_nyse_stocks.py`.

It will output a graph of the actual stock prices and the predicted stock prices.

## Other things

+ Currently, the network easily gets overtrained if you train it until the `early-stopping` module stops the training process.


+ Other files not mentioned in this readme are weights of previous trainings, and has directly nothing to do with the network.
