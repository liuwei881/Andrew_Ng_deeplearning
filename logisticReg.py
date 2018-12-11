# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

index = 25
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y_orig[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y_orig[:, index])].decode("utf-8") + "' picture.")


### START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
### END CODE HERE ###

print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y_orig.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y_orig.shape))


####
#A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b∗∗c∗∗d, a)
# is to use: X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X
# 拉伸4维的矩阵(a, b, c, d) to (b**c**d, a)使用 X.reshape(X.shape[0], -1).T
####
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y_orig.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y_orig.shape))
print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

#### 图片拉伸后执行标准归一化处理
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

####对于图片数据
#Common steps for pre-processing a new dataset are:
# 先划分数据集训练集与测试集
#- Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, …)
# 将图片拉伸
#- Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
# 标准化数据
#- “Standardize” the data


