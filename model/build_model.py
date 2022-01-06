"""
 UniDirectional LSTM Model 4 kinds
 KCSE 2021
"""

from keras.models import Model
from keras.layers import Input, Dense, LSTM
import tensorflow as tf
import flag
import numpy as np
"""
 Build Option
 1)
 - Many to One (M2O)
 - Many to Many (M2M)

 2)
 - Multi layered
 - Stacked
"""


# def LSTM_Model_M2O_ML(x, y, test_x, test_y, n_cell, n_steps, n_features):
#     lstm_input = Input(batch_shape=(None, n_steps, n_features))
#     lstm_layer = LSTM(n_cell, activation='sigmoid')(lstm_input)
#     lstm_output = Dense(1)(lstm_layer)
#
#     model = Model(lstm_input, lstm_output)
#     model.compile(
#         loss='mean_squared_error',
#         optimizer='adam'
#     )
#     model.fit(x, y, epochs=50, validation_data=[test_x, test_y], batch_size=flag.BATCH_SIZE, verbose=0)
#     model.predict(x, batch_size=flag.BATCH_SIZE)
#
#
# def LSTM_Model_M2M_ML(x, y, test_x, test_y, n_cell, n_steps, n_features):
#     lstm_input = Input(batch_shape=(None, n_steps, n_features))
#     lstm_layer = LSTM(n_cell, activation='sigmoid', return_sequences=True)(lstm_input)
#     lstm_output = TimeDistributed(Dense(1))(lstm_layer)
#
#     model = Model(lstm_input, lstm_output)
#     model.compile(
#         loss='mean_squared_error',
#         optimizer='adam'
#     )
#
#     model.fit(x, y, epochs=50, validation_data=[test_x, test_y],batch_size=flag.BATCH_SIZE, verbose=0)
#     model.predict(x, batch_size=flag.BATCH_SIZE)


def LSTM_Model_M2O_Stacked(x, y, test_x, test_y, n_cell):
    print("-- LSTM_Model_M2O_Stacked --")
    print(np.shape(x))
    print(np.shape(y))
    lstm_input = Input(batch_shape=(flag.BATCH_SIZE, np.shape(x)[1], np.shape(x)[2]))
    lstm_layer_1 = LSTM(n_cell, activation='tanh', return_sequences=True)(lstm_input)
    lstm_layer_2 = LSTM(n_cell, activation='tanh', return_sequences=True)(lstm_layer_1)
    lstm_output = Dense(1)(lstm_layer_2)

    model = Model(lstm_input, lstm_output)
    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    model.summary()
    model.fit(x,
              y,
              batch_size=flag.BATCH_SIZE,
              validation_data=(test_x, test_y),
              validation_batch_size=(flag.BATCH_SIZE, np.shape(test_x)[1], np.shape(test_x)[2]),
              epochs=200, verbose=1)
    model.evaluate(test_x, test_y, batch_size=flag.BATCH_SIZE)
    # model.predict(test_x, batch_size=flag.BATCH_SIZE)


# def LSTM_Model_M2M_Stacked(x, y, test_x, test_y, n_cell, n_steps, n_features):
#     lstm_input = Input(batch_shape=(None, n_steps, n_features))
#     lstm_layer_1 = LSTM(n_cell, activation='tanh', return_sequences=True)(lstm_input)
#     lstm_layer_2 = LSTM(n_cell, activation='tanh')(lstm_layer_1)
#     lstm_output = TimeDistributed(Dense(1))(lstm_layer_2)
#
#     model = Model(lstm_input, lstm_output)
#     model.compile(
#         loss='mean_squared_error',
#         optimizer='adam'
#     )
#
#     model.fit(x, y, validation_data=[test_x, test_y], epochs=50, batch_size=flag.BATCH_SIZE, verbose=0)
#     model.predict(x, batch_size=flag.BATCH_SIZE)

