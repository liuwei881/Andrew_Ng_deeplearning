# coding=utf-8

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)

# y_hat = tf.constant(36, name='y_hat')
# y = tf.constant(39, name='y')
#
# loss = tf.Variable((y - y_hat)**2, name='loss')
# init = tf.global_variables_initializer()
#
# with tf.Session() as session:
#     session.run(init)
#     print(session.run(loss))
#
# with tf.Session() as session:
#     a = tf.constant(2)
#     b = tf.constant(10)
#     c = tf.multiply(a, b)
#     print(session.run(c))
#
#
# #   总结:
# #   remember to initialize your variables,
# #   create a session and run the operations inside the session
#
# with tf.Session() as session:
#     x = tf.placeholder(tf.int64, name='x')
#     print(session.run(2 * x, feed_dict={x: 3}))


def linear_function():
    """
    Implements a linear function:
        Initializes W to be a random tensor of shape (4,3)
        Initializes X to be a random tensor of shape (3,1)
        Initializes b to be a random tensor of shape (4,1)
    Returns:
        result -- runs the session for Y = WX + b
    """
    np.random.seed(1)
    X = tf.constant(np.random.randn(3, 1), name='X')
    W = tf.constant(np.random.randn(4, 3), name='W')
    b = tf.constant(np.random.randn(4, 1), name='b')
    Y = tf.add(tf.matmul(W, X), b)
    with tf.Session() as session:
        result = session.run(Y)
    return result


# print("result = " + str(linear_function()))

def sigmoid(z):
    """
    Computes the sigmoid of z
        Arguments:
        z -- input value, scalar or vector
    Returns:
        results -- the sigmoid of z
    """
    x = tf.placeholder(tf.float32, name='x')
    s = tf.sigmoid(x)
    with tf.Session() as session:
        result = session.run(s, feed_dict={x: z})
    return result


# print("sigmoid(0) = " + str(sigmoid(0)))
# print("sigmoid(12) = " + str(sigmoid(12)))


# To summarize, you how know how to:
# 1. Create placeholders
# 2. Specify the computation graph corresponding to operations you want to compute
# 3. Create the session
# 4. Run the session, using a feed dictionary if necessary to specify placeholder variables’ values.


def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
        Arguments:
        logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
        labels -- vector of labels y (1 or 0)
        Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels"
        in the TensorFlow documentation. So logits will feed into z, and labels into y.
    Returns:
        cost -- runs the session of the cost (formula (2))
    """
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)
    with tf.Session() as session:
        result = session.run(cost, feed_dict={z: logits, y: labels})
    return result


# logits = sigmoid(np.array([0.2, 0.4, 0.7, 0.9]))
# cost = cost(logits, np.array([0, 0, 1, 1]))
# print("cost = " + str(cost))


def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
    corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
    will be 1.
    Arguments:
        labels -- vector containing the labels
        C -- number of classes, the depth of the one hot dimension

    Returns:
        one_hot -- one hot matrix
    """
    C = tf.constant(value=C, name='C')
    with tf.Session() as session:
        one_hot = session.run(tf.one_hot(labels, C, axis=0))
    return one_hot


# labels = np.array([1, 2, 3, 0, 2, 1])
# one_hot = one_hot_matrix(labels, C=4)
# print("one_hot = " + str(one_hot))


def ones(shape):
    """
    Creates an array of ones of dimension shape
    Arguments:
        shape -- shape of the array you want to create

    Returns:
        ones -- array containing only ones
        """
    ones = tf.ones(shape)
    with tf.Session() as session:
        ones = session.run(ones)
    return ones


# print("ones = " + str(ones([3])))
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# 拉伸图片
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# 图片归一化
X_train = X_train_flatten/255
X_test = X_test_flatten/255

Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

# print("number of training examples = " + str(X_train.shape[1]))
# print("number of test examples = " + str(X_test.shape[1]))
# print("X_train shape: " + str(X_train.shape))
# print("Y_train shape: " + str(Y_train.shape))
# print("X_test shape: " + str(X_test.shape))
# print("Y_test shape: " + str(Y_test.shape))


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
        n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
        n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
        X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
        Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
        - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
          In fact, the number of examples during test/train is different.
    """
    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])
    return X, Y


# X, Y = create_placeholders(12288, 6)
# print("X = " + str(X))
# print("Y = " + str(Y))


def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                           W1 : [25, 12288]
                           b1 : [25, 1]
                           W2 : [12, 25]
                           b2 : [12, 1]
                           W3 : [6, 12]
                           b3 : [6, 1]

    Returns:
       parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    tf.set_random_seed(1)
    W1 = tf.get_variable('W1', [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1', [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2', [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2', [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3', [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3', [6, 1], initializer=tf.zeros_initializer())
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    return parameters


# tf.reset_default_graph()
# with tf.Session() as sess:
#     parameters = initialize_parameters()
#     print("W1 = " + str(parameters["W1"]))
#     print("b1 = " + str(parameters["b1"]))
#     print("W2 = " + str(parameters["W2"]))
#     print("b2 = " + str(parameters["b2"]))


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                      the shapes are given in initialize_parameters
    Returns:
        Z3 -- the output of the last LINEAR unit
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2 , A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    return Z3


# tf.reset_default_graph()
#
# with tf.Session() as sess:
#     X, Y = create_placeholders(12288, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     print("Z3 = " + str(Z3))


def compute_cost(Z3, Y):
    """
    Computes the cost
    Arguments:
        Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
        Y -- "true" labels vector placeholder, same shape as Z3
    Returns:
        cost - Tensor of the cost function
    """
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    return cost


# tf.reset_default_graph()
#
# with tf.Session() as sess:
#     X, Y = create_placeholders(12288, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3, Y)
#     print("cost = " + str(cost))

# For instance, for gradient descent the optimizer would be:
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
# To make the optimization you would do:
# _ , c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    Arguments:
        X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
        Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
        X_test -- training set, of shape (input size = 12288, number of training examples = 120)
        Y_test -- test set, of shape (output size = 6, number of test examples = 120)
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs
    Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph() # 可以在不覆盖tf变量的情况下重新运行模型.
    tf.set_random_seed(1) # 随机数种子
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    epoch_cost = 0
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m/minibatch_size)
            seed = seed+1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters


parameters = model(X_train, Y_train, X_test, Y_test)
