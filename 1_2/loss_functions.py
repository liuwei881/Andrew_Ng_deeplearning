# coding=utf-8

import numpy as np


def L1(yhat, y):
    """
    定义L1正则
    :param yhat: 预测值
    :param y: 真实值
    :return:
    """
    loss = np.sum(np.abs(yhat - y))
    return loss


def L2(yhat, y):
    """
    定义L2正则
    :param yhat:
    :param y:
    :return:
    """
    loss = np.sum(np.power((yhat - y), 2))
    return loss


yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L2(yhat, y)))