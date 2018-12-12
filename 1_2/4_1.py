# coding=utf-8

# The main steps for building a Neural Network are:
# 1. Define the model structure (such as number of input features)
# 2. Initialize the model’s parameters
# 3. Loop:
# - Calculate current loss (forward propagation)
# - Calculate current gradient (backward propagation)
# - Update parameters (gradient descent)
#
# You often build 1-3 separately and integrate them into one function we call model().

import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255


def sigmod(z):
    """
    sigmod 函数
    :param x: 输入值
    :return: 输出0, 1的值
    """
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    # For image inputs, w will be of shape (num_px × num_px × 3, 1).
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, X, Y):
    """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b

        Tips:
        - Write your code step by step for the propagation. np.log(), np.dot()
        Forward Propagation:
        - You get X
        - You compute A=σ(wTX+b)=(a(0),a(1),...,a(m−1),a(m))A=σ(wTX+b)=(a(0),a(1),...,a(m−1),a(m))
        - You calculate the cost function: J=−1m∑mi=1y(i)log(a(i))+(1−y(i))log(1−a(i))J=−1m∑i=1my(i)log⁡(a(i))+(1−y(i))log⁡(1−a(i))
        Here are the two formulas you will be using:

        ∂J/∂w=1m*X(A−Y)T
        ∂J/∂b=1/m∑(a(i)−y(i)

    """
    m = X.shape[1]
    # 前项传播
    A = sigmod(np.dot(w.T, X) + b)
    cost = -(1.0/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

    # 反向传播
    dw = (1.0/m) * np.dot(X, (A-Y).T)
    db = (1.0/m) * np.sum(A-Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {'dw': dw, 'db': db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
       This function optimizes w and b by running a gradient descent algorithm

       Arguments:
       w -- weights, a numpy array of size (num_px * num_px * 3, 1)
       b -- bias, a scalar
       X -- data of shape (num_px * num_px * 3, number of examples)
       Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
       num_iterations -- number of iterations of the optimization loop
       learning_rate -- learning rate of the gradient descent update rule
       print_cost -- True to print the loss every 100 steps

       Returns:
       params -- dictionary containing the weights w and bias b
       grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
       costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

       Tips:
       You basically need to write down two steps and iterate through them:
           1) Calculate the cost and the gradient for the current parameters. Use propagate().
           2) Update the parameters using gradient descent rule for w and b.
       """
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration {0}: {1}".format(i, cost))

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}
    return params, grads, costs


def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """
    # You’ve implemented several functions that:
    # - Initialize(w, b)
    # - Optimize the loss iteratively to learn parameters(w, b):
    # - computing the cost and its gradient
    # - updating the parameters using gradient descent
    # - Use the learned(w, b) to predict the labels for a given set of examples
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmod(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    assert(Y_prediction.shape == (1, m))

    return Y_prediction


# w = np.array([[0.1124579], [0.23106775]])
# b = -0.3
# X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
# print("predictions = " + str(predict(w, b, X)))
# w, b, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
# params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """
    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters['w']
    b = parameters['b']

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


# In order for Gradient Descent to work you must choose the learning rate wisely.
# The learning rate αα determines how rapidly we update the parameters.
# If the learning rate is too large we may “overshoot” the optimal value.
# Similarly, if it is too small we will need too many iterations to converge to the best values.
# That’s why it is crucial to use a well-tuned learning rate.
# Let’s compare the learning curve of our model with several choices of learning rates.
# Run the cell below. This should take about 1 minute.
# Feel free also to try different values than the three we have initialized the learning_rates variable to contain, and see what happens.

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


# Interpretation:
# - Different learning rates give different costs and thus different predictions results.
# - If the learning rate is too large (0.01), the cost may oscillate up and down. It may even diverge (though in this example, using 0.01 still eventually ends up at a good value for the cost).
# - A lower cost doesn’t mean a better model. You have to check if there is possibly overfitting. It happens when the training accuracy is a lot higher than the test accuracy.
# - In deep learning, we usually recommend that you:
# - Choose the learning rate that better minimizes the cost function.
# - If your model overfits, use other techniques to reduce overfitting. (We’ll talk about this in later videos.)
