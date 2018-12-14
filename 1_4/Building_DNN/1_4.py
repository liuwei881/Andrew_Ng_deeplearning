# coding=utf-8

import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v3 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# Initialize the parameters for a two-layer network and for an LL-layer neural network.
# Implement the forward propagation module (shown in purple in the figure below).
# Complete the LINEAR part of a layer’s forward propagation step (resulting in Z[l]Z[l]).
# We give you the ACTIVATION function (relu/sigmoid).
# Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
# Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer LL).
# This gives you a new L_model_forward function.
# Compute the loss.
# Implement the backward propagation module (denoted in red in the figure below).
# Complete the LINEAR part of a layer’s backward propagation step.
# We give you the gradient of the ACTIVATE function (relu_backward/sigmoid_backward)
# Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
# Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
# Finally update the parameters.


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# parameters = initialize_parameters(3, 2, 1)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


# Instructions:
# - The model’s structure is [LINEAR -> RELU] ×× (L-1) -> LINEAR -> SIGMOID.
# I.e., it has L−1L−1 layers using a ReLU activation function followed by an output layer with a sigmoid activation function.
# - Use random initialization for the weight matrices. Use np.random.randn(shape) * 0.01.
# - Use zeros initialization for the biases. Use np.zeros(shape).
# - We will store n[l]n[l], the number of units in different layers, in a variable layer_dims.
# For example, the layer_dims for the “Planar Data classification model” from last week would have been [2,4,1]: There were two inputs, one hidden layer with 4 hidden units, and an output layer with 1 output unit.
# Thus means W1’s shape was (4,2), b1 was (4,1), W2 was (1,4) and b2 was (1,1). Now you will generalize this to LL layers!
# - Here is the implementation for L=1L=1 (one layer neural network). It should inspire you to implement the general case (L-layer neural network).


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters


# parameters = initialize_parameters_deep([5, 4, 3])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


# Now that you have initialized your parameters, you will do the forward propagation module.
# You will start by implementing some basic functions that you will use later when implementing the model.
# You will complete three functions in this order:
#
# LINEAR
# LINEAR -> ACTIVATION where ACTIVATION will be either ReLU or Sigmoid.
# [LINEAR -> RELU] ×× (L-1) -> LINEAR -> SIGMOID (whole model)
# The linear forward module (vectorized over all the examples) computes the following equations:
#
# Z[l]=W[l]A[l−1]+b[l](4)
# (4)Z[l]=W[l]A[l−1]+b[l]
# where A[0]=XA[0]=X.
#
# Exercise: Build the linear part of forward propagation.
#
# Reminder:
# The mathematical representation of this unit is Z[l]=W[l]A[l−1]+b[l]Z[l]=W[l]A[l−1]+b[l].
# You may also find np.dot() useful. If your dimensions don’t match, printing W.shape may help.


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache


# A, W, b = linear_forward_test_case()
# # Z, linear_cache = linear_forward(A, W, b)
# # print("Z = " + str(Z))

# In this notebook, you will use two activation functions:
# Sigmoid: σ(Z)=σ(WA+b)=11+e−(WA+b)σ(Z)=σ(WA+b)=11+e−(WA+b).
# We have provided you with the sigmoid function.
# This function returns two items: the activation value “a” and a “cache” that contains “Z” (it’s what we will feed in to the corresponding backward function). To use it you could just call:
# A, activation_cache = sigmoid(Z)
# 1
# ReLU: The mathematical formula for ReLu is A=RELU(Z)=max(0,Z)A=RELU(Z)=max(0,Z).
# We have provided you with the relu function.
# This function returns two items: the activation value “A” and a “cache” that contains “Z” (it’s what we will feed in to the corresponding backward function). To use it you could just call:
# A, activation_cache = relu(Z)
# 1
# For more convenience, you are going to group two functions (Linear and Activation) into one function (LINEAR->ACTIVATION).
# Hence, you will implement a function that does the LINEAR forward step followed by an ACTIVATION forward step.
#
# Exercise: Implement the forward propagation of the LINEAR->ACTIVATION layer. Mathematical relation is: A[l]=g(Z[l])=g(W[l]A[l−1]+b[l])A[l]=g(Z[l])=g(W[l]A[l−1]+b[l]) where the activation “g” can be sigmoid() or relu(). Use linear_forward() and the correct activation function.


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache


# A_prev, W, b = linear_activation_forward_test_case()
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="sigmoid")
# print("With sigmoid: A = " + str(A))
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="relu")
# print("With ReLU: A = " + str(A))


# Exercise: Implement the forward propagation of the above model.
# Instruction: In the code below, the variable AL will denote A[L]=σ(Z[L])=σ(W[L]A[L−1]+b[L])A[L]=σ(Z[L])=σ(W[L]A[L−1]+b[L]).
# (This is sometimes also called Yhat, i.e., this is Y^Y^.)
#
# Tips:
# - Use the functions you had previously written
# - Use a for loop to replicate [LINEAR->RELU] (L-1) times
# - Don’t forget to keep track of the caches in the “caches” list. To add a new value c to a list, you can use list.append(c).


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
    assert(AL.shape == (1, X.shape[1]))
    return AL, caches


X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))




