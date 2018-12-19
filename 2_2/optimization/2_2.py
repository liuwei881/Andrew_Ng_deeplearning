# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *


def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return parameters


parameters, grads, learning_rate = update_parameters_with_gd_test_case()

parameters = update_parameters_with_gd(parameters, grads, learning_rate)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# 批量梯度下降
# X = data_input
# Y = labels
# parameters = initialize_parameters(layers_dims)
# for i in range(0, num_iterations):
#     a, caches = forward_propagation(X, parameters)
#     cost = compute_cost(a, Y)
#     grads = backward_propagation(a, caches, parameters)
#     parameters = update_parameters(parameters, grads)
#随机梯度(一个样本)下降
# X = data_input
# Y = labels
# parameters = initialize_parameters(layers_dims)
# for i in range(0, num_iterations):
#     for j in range(0, m):
#         a, caches = forward_propagation(X[:,j], parameters)
#         cost = compute_cost(a, Y[:, j])
#         grads = backward_propagation(a, caches, parameters)
#         parameters = update_parameters(parameters, grads)