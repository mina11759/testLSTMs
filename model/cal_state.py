# for coverage

from keras.models import Model
import keras.backend as K
import numpy as np

"""
input layer
hidden layer
output layer
"""


def hard_sigmoid(x):
    return np.maximum(0, np.minimum(1, 0.2*x + 0.5))


def _evaluate(model: Model, nodes_to_evaluate, x, y=None):
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, nodes_to_evaluate)
    x_, y_, sample_weight_ = model._standardize_user_data(x, y)
    return f(x_ + y_ + sample_weight_)


def get_activatinos_single_layer(model, x, layer_name=None):
    nodes = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is Nome]
    input_layer_outputs , layer_outputs = [], []
    [input_layer_outputs.append(node) if 'input_' in node.name else layer_outputs.append(node) for node in nodes]
    activations = _evaluate(model, layer_outputs, x, y=None)
    activations_dict = dict(zip([output.name for output in layer_outputs], activations))
    activations_inputs_dict = dict(zip([output.name for output in input_layer_outputs]))
    result = activations_inputs_dict.copy()
    result.update(activations_dict)
    return np.squeeze(list(result.values())[0])


def cal_lstm_state(model, test, layer_num):

    units = int()

    W = model.layers[layer_num].get_weight()[0]
    U = model.layers[layer_num].get_weight()[1]
    b = model.layers[layer_num].get_weight()[2]

    W_i = W[:, :units]
    W_f = W[:, units: units * 2]
    W_c = W[:, units * 2: units * 3]
    W_o = W[:, units * 3:]

    U_i = W[:, :units]
    U_f = W[:, units: units * 2]
    U_c = W[:, units * 2: units * 3]
    U_o = W[:, units * 3:]

    b_i = W[:, :units]
    b_f = W[:, units: units * 2]
    b_c = W[:, units * 2: units * 3]
    b_o = W[:, units * 3:]

    # calculate
    h_t = np.zeros((temp, units))
    c_t = np.zeros((temp, units))
    f_t = np.zeros((temp, units))
    h_t0 = np.zeros((1, units))
    c_t0 = np.zeros((1, units))

    for i in range(0, temp):
        f_gate = hard_sigmoid(np.dot)