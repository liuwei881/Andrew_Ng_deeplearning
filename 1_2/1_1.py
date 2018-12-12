# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt


def sigmod(x):
    """
    sigmod函数
    :param x:
    :return:
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """
    sigmod 函数的导数
    :param x:
    :return:
    """
    ds = sigmod(x) * (1 - sigmod(x))
    return ds


def image2vector(image):
    """
    拉伸一张图片
    :param image:
    :return:
    """
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))
    return v


def normalizeRows(x):
    """
    归一化向量 x_norm 向量的一范数, axis=0表示按列去, axis=1表示按行取范数
    :param x:
    :return:
    """
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x


def softmax(x):
    """
    softmax函数
    :param x:
    :return:
    """
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s


if __name__ == '__main__':
    x = np.array([[0, 3, 4], [1, 6, 4]])
    print(softmax(x))
    # print(normalizeRows(x))
    # x = np.arange(-5.0, 5.0, 0.2)
    # y = sigmoid_derivative(x)
    # print(y)
    # plt.plot(x, y, color='green')
    # plt.show()
    # image = np.array([[[ 0.67826139,  0.29380381],
    #     [0.90714982,  0.52835647],
    #     [0.4215251,  0.45017551]],
    #
    #    [[0.92814219,  0.96677647],
    #     [0.85304703,  0.52351845],
    #     [0.19981397,  0.27417313]],
    #
    #    [[0.60659855,  0.00533165],
    #     [0.10820313,  0.49978937],
    #     [0.34144279,  0.94630077]]])
    # print(image2vector(image))
