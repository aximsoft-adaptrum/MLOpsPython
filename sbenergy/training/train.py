"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import os
import numpy as np
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt

np.random.seed(1234)
tf.random.set_seed(1234)

def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon

    for i in range(start, end):
        indices = range(i-window, i)
        X.append(dataset[indices])

        indicey = range(i+1, i+1+horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y)

def split_data(df, horizon, hist_window):
    X_data = df[['Direct Normal Irradiance (DNI) W/m2']].values
    Y_data = df[['Direct Normal Irradiance (DNI) W/m2']].values
    TRAIN_SPLIT = round(len(df)*(0.7))
    X_train, y_train = custom_ts_multi_data_prep(X_data, Y_data, 0, TRAIN_SPLIT, hist_window, horizon)
    X_test, y_test = custom_ts_multi_data_prep(X_data, Y_data, TRAIN_SPLIT, None, hist_window, horizon)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    return data

def model_initialize(data, horizon):
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=data["train"]["X"].shape[-2:]))                
    lstm_model.add(Dense(horizon))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    return lstm_model

def train_model(model, data, epochs):
    model.fit(data["train"]["X"], data["train"]["y"],batch_size=10, epochs=epochs,validation_data=(data["test"]["X"],data["test"]["y"]),verbose=1,shuffle=False)
    return model

def get_model_metrics(model, data, val):
    X_data1 = data[['Direct Normal Irradiance (DNI) W/m2']][-35:].values
    pred_list = []
    for i in range(25): 
        data_val = X_data1[-35+i:-25+i]
        val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])
        pred = model.predict(val_rescaled)
        pred_list.append(pred)
    a = [l.tolist() for l in pred_list]
    pred_dataframe = pd.DataFrame(val['Direct Normal Irradiance (DNI) W/m2'])
    a = np.array(a)
    a = a.reshape(25,)
    pred_dataframe['pred'] = a
    mse = mean_squared_error(pred_dataframe['Direct Normal Irradiance (DNI) W/m2'], pred_dataframe['pred'])
    metrics = {"mse": mse}
    return metrics

def main():
    print("Running train.py")
    hist_window = 10
    horizon = 1
    # Define training parameters
    epochs = 10
    # Load the training data as dataframe
    data_dir = "data"
    data_file = os.path.join(data_dir, 'juno_3month_one_hour.csv')
    train_df_main = pd.read_csv(data_file, index_col=0)
    train_df = pd.read_csv(data_file, index_col=0)
    train_df.index = pd.to_datetime(train_df.index)
    val = train_df.tail(25)
    data = train_df.drop(train_df['Direct Normal Irradiance (DNI) W/m2'][-25:].index)

    data = split_data(data, horizon, hist_window)

    #model initialize
    initialize_model = model_initialize(data, horizon)

    # Train the model
    model = train_model(initialize_model, data, epochs)

    # Log the metrics for the model
    metrics = get_model_metrics(model, train_df, val)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")

if __name__ == '__main__':
    main()