import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import Model
import numpy as np

from model.model_manager import ModelManager


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


class Temperature(ModelManager):
    def __init__(self, model_name):
        super().__init__(model_name)

    def get_intermediate_output(self, layer, data):
        intermediate_layer_model = Model(self.model.input, self.model.get_layer(layer.name).output)
        return intermediate_layer_model.predict(np.expand_dims(data, axis=0))

    def load_model(self):
        self.model: Model = load_model('models/' + self.model_name + '.h5',
                                custom_objects={'root_mean_squared_error': root_mean_squared_error}, compile=False)
        optimizer = Adadelta(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error',
                           metrics=[root_mean_squared_error])
        self.model.summary()

    def train_model_M2O_Stacked(self, x_train, y_train, x_test, y_test):
        n_hidden = 64
        n_seq = 12 # time step
        n_input = 6 # n_feature
        n_output = 1
        epochs= 150
        batch_size = 32

        input_layer = tf.keras.Input(shape=(n_seq, n_input))
        lstm1 = LSTM(n_hidden, return_sequences=True)(input_layer)
        lstm2 = LSTM(n_hidden, activation='tanh')(lstm1)
        output = Dense(n_output * 4)(lstm2)
        output = Dense(n_output, activation='linear')(output)

        model = Model(input_layer, output)
        opt = Adadelta(lr=0.001)
        model.compile(optimizer=opt, loss='mean_squared_error',
                      metrics=[root_mean_squared_error])

        model.fit(x=x_train, y=y_train,
                  validation_data=(x_test, y_test),
                  batch_size=batch_size, epochs=epochs, shuffle=True)

        model.save('models/%s.h5' % self.model_name)
        self.model = model

    def train_model_M2O_Multilayered(self, x_train, y_train, x_test, y_test):
        n_hidden = 64
        n_seq = 12 # time step
        n_input = 6 # n_feature
        n_output = 1
        epochs= 150
        batch_size = 32

        input_layer = tf.keras.Input(shape=(n_seq, n_input))
        lstm1 = LSTM(n_hidden * 2, return_sequences=True)(input_layer)
        # lstm2 = LSTM(n_hidden, activation='tanh')(lstm1)
        output = Dense(n_output * 4)(lstm1)
        output = Dense(n_output, activation='linear')(output)

        model = Model(input_layer, output)
        opt = Adadelta(lr=0.001)
        model.compile(optimizer=opt, loss='mean_squared_error',
                      metrics=[root_mean_squared_error])

        model.fit(x=x_train, y=y_train,
                  validation_data=(x_test, y_test),
                  batch_size=batch_size, epochs=epochs, shuffle=True)

        model.save('models/%s.h5' % self.model_name)
        self.model = model

    def train_model_M2M_Stacked(self, x_train, y_train, x_test, y_test):
        n_hidden = 64
        n_seq = 12 # time step
        n_input = 6 # n_feature
        n_output = 6
        epochs= 150
        batch_size = 32

        input_layer = tf.keras.Input(shape=(n_seq, n_input))
        lstm1 = LSTM(n_hidden, return_sequences=True)(input_layer)
        lstm2 = LSTM(n_hidden, activation='tanh')(lstm1)
        output = Dense(n_output * 4)(lstm2)
        output = Dense(n_output, activation='linear')(output)

        model = Model(input_layer, output)
        opt = Adadelta(lr=0.001)
        model.compile(optimizer=opt, loss='mean_squared_error',
                      metrics=[root_mean_squared_error])

        model.fit(x=x_train, y=y_train,
                  validation_data=(x_test, y_test),
                  batch_size=batch_size, epochs=epochs, shuffle=True)

        model.save('models/%s.h5' % self.model_name)
        self.model = model

    def train_model_M2M_Multilayered(self, x_train, y_train, x_test, y_test):
        n_hidden = 64
        n_seq = 12 # time step
        n_input = 6 # n_feature
        n_output = 6
        epochs= 150
        batch_size = 32

        input_layer = tf.keras.Input(shape=(n_seq, n_input))
        lstm1 = LSTM(n_hidden * 2, return_sequences=True)(input_layer)
        # lstm2 = LSTM(n_hidden, activation='tanh')(lstm1)
        output = Dense(n_output * 4)(lstm1)
        output = Dense(n_output, activation='linear')(output)

        model = Model(input_layer, output)
        opt = Adadelta(lr=0.001)
        model.compile(optimizer=opt, loss='mean_squared_error',
                      metrics=[root_mean_squared_error])

        model.fit(x=x_train, y=y_train,
                  validation_data=(x_test, y_test),
                  batch_size=batch_size, epochs=epochs, shuffle=True)

        model.save('models/%s.h5' % self.model_name)
        self.model = model

    def test_model(self, test_x, test_y):
        return self.model.evaluate(test_x, test_y)[1]

    def get_layer(self, index):
        return self.model.layers[index]

    @staticmethod
    def __get_layer_type(layer_name):
        return layer_name.split('_')[0]

    def get_lstm_layer(self):
        indices = []
        layers = []
        for index, layer in enumerate(self.model.layers):
            if 'input' in layer.name \
                    or 'concatenate' in layer.name \
                    or index == len(self.model.layers) - 1 \
                    or 'flatten' in layer.name:
                continue
            layer_type = self.__get_layer_type(layer.name)
            if layer_type == "lstm":
                layers.append(layer)
                indices.append(index)
        return indices, layers

    def get_fc_layer(self):
        indices = []
        layers = []
        for index, layer in enumerate(self.model.layers):
            if 'input' in layer.name \
                    or 'concatenate' in layer.name \
                    or index == len(self.model.layers) - 1 \
                    or 'flatten' in layer.name:
                continue
            layer_type = self.__get_layer_type(layer.name)
            if layer_type == "dense":
                layers.append(layer)
                indices.append(index)
        return indices, layers

    def get_prob(self, data):
        data = data[np.newaxis, :]
        prob = np.squeeze(self.model.predict(data))
        return prob