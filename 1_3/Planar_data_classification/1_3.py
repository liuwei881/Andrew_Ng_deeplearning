# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
np.random.seed(1) # set a seed so that the results are consistent


X, Y = load_planar_dataset()

# plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral)
# plt.show()

shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

LR_predictions = clf.predict(X.T)

print('Accuracy of logistic regression: %d ' % float((np.dot(Y, LR_predictions) + np.dot(1-Y, 1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# Reminder: The general methodology to build a Neural Network is to:
# 1. Define the neural network structure ( # of input units, # of hidden units, etc).
# 2. Initialize the model’s parameters
# 3. Loop:
# - Implement forward propagation
# - Compute loss
# - Implement backward propagation to get the gradients
# - Update parameters (gradient descent)


def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y


# X_assess, Y_assess = layer_sizes_test_case()
# n_x, n_h, n_y = layer_sizes(X_assess, Y_assess)
# print(n_x, n_h, n_y)

# Instructions:
# - Make sure your parameters’ sizes are right. Refer to the neural network figure above if needed.
# - You will initialize the weights matrices with random values.
# - Use: np.random.randn(a,b) * 0.01 to randomly initialize a matrix of shape (a,b).
# - You will initialize the bias vectors as zeros.
# - Use: np.zeros((a,b)) to initialize a matrix of shape (a,b) with zeros.


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
    W1 -- weight matrix of shape (n_h, n_x)(4, 2)
    b1 -- bias vector of shape (n_h, 1)(4, 1)
    W2 -- weight matrix of shape (n_y, n_h)(2, 4)
    b2 -- bias vector of shape (n_y, 1)(2, 1)
    """
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# n_x, n_h, n_y = initialize_parameters_test_case()

# parameters = initialize_parameters(n_x, n_h, n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# Instructions:
# - Look above at the mathematical representation of your classifier.
# - You can use the function sigmoid(). It is built-in (imported) in the notebook.
# - You can use the function np.tanh(). It is part of the numpy library.
# - The steps you have to implement are:
# 1. Retrieve each parameter from the dictionary “parameters” (which is the output of initialize_parameters()) by using parameters[".."].
# 2. Implement Forward Propagation. Compute Z[1],A[1],Z[2],Z[1],A[1],Z[2] and A[2]A[2]
# (the vector of all your predictions on all the examples in the training set).
# - Values needed in the backpropagation are stored in “cache“.
# The cache will be given as an input to the backpropagation function.


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

    return A2, cache


# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)
# print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

# Instructions:
# - There are many ways to implement the cross-entropy loss.
# To help you, we give you how we would have implemented
# −∑i=0my(i)log(a[2](i))−∑i=0my(i)log⁡(a[2](i)):
#
# logprobs = np.multiply(np.log(A2),Y)
# cost = - np.sum(logprobs)                # no need to use a for loop!


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), (1-Y))
    cost = -(1.0 / m) * np.sum(logprobs)

    cost = np.squeeze(cost)
    assert(isinstance(cost, float))
    return cost


# A2, Y_assess, parameters = compute_cost_test_case()
# print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    Returns:

    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = 1.0/m*np.dot(dZ2, A1.T)
    db2 = 1.0/m*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1-np.power(A1, 2))
    dW1 = 1.0/m*np.dot(dZ1, X.T)
    db1 = 1.0/m*np.sum(dZ1, axis=1, keepdims=True)

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads


# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
#
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print("dW1 = "+ str(grads["dW1"]))
# print("db1 = "+ str(grads["db1"]))
# print("dW2 = "+ str(grads["dW2"]))
# print("db2 = "+ str(grads["db2"]))


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learn by the model. They can then be used to predict.
    """
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate=1.2)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {0}: {1}".format(i, cost))
    return parameters


# X_assess, Y_assess = nn_model_test_case()
# parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    return predictions


parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

# Plot the decision boundary
# plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
# plt.title("Decision Boundary for hidden layer size " + str(4))
# plt.show()

predictions = predict(parameters, X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T)) / float(Y.size)*100) + '%')

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 10]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T)) / float(Y.size)*100)
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))


# Interpretation:
# - The larger models (with more hidden units) are able to fit the training set better,
# until eventually the largest models overfit the data.
# - The best hidden layer size seems to be around n_h = 5.
# Indeed, a value around here seems to fits the data well without also incurring noticable overfitting.
# - You will also learn later about regularization,
# which lets you use very large models (such as n_h = 50) without much overfitting.

# Optional questions:
#
# Note: Remember to submit the assignment but clicking the blue “Submit Assignment” button at the upper-right.
#
# Some optional/ungraded questions that you can explore if you wish:
# - What happens when you change the tanh activation for a sigmoid activation or a ReLU activation?
# - Play with the learning_rate. What happens?
# - What if we change the dataset? (See part 5 below!)

# noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
#
# datasets = {"noisy_circles": noisy_circles,
#             "noisy_moons": noisy_moons,
#             "blobs": blobs,
#             "gaussian_quantiles": gaussian_quantiles}
#
# dataset = "noisy_moons"
#
# X, Y = datasets[dataset]
# X, Y = X.T, Y.reshape(1, Y.shape[0])
#
# if dataset == "blobs":
#     Y = Y % 2
# plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral)
# plt.show()
