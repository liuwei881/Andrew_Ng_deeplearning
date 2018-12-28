# coding=utf-8

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

np.random.seed(1)


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()