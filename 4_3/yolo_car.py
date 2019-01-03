# coding=utf-8
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body


def iou(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2
    Arguments:
        box1 -- first box, list object with coordinates (x1, y1, x2, y2)
        box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """
    # 取交集
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    # 取并集
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    # 交集与并集比
    iou = inter_area / union_area
    return iou


# box1 = (2, 1, 4, 3)
# box2 = (1, 2, 3, 4)
# print("iou = " + str(iou(box1, box2)))
# 非最大值抑制三个步骤
# 选择高分数的框
# 1. Select the box that has the highest score.
# 计算重叠的框，删除比iou_threshold阈值小的框
# 2. Compute its overlap with all other boxes, and remove boxes that overlap it more than iou_threshold.
# 返回1,2 迭代直到没有比当前所选框更低的
# 3. Go back to step 1 and iterate until there’s no more boxes with a lower score than the current selected box.
# 这将选择最好的框
# This will remove all boxes that have a large overlap with the selected boxes. Only the “best” boxes remain.


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
        scores -- tensor of shape (None,), output of yolo_filter_boxes()
        boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
        classes -- tensor of shape (None,), output of yolo_filter_boxes()
        max_boxes -- integer, maximum number of predicted boxes you'd like
        iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
        scores -- tensor of shape (, None), predicted score for each box
        boxes -- tensor of shape (4, None), predicted box coordinates
        classes -- tensor of shape (, None), predicted class for each box

        Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
        function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold, name=None)
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    return scores, boxes, classes


# with tf.Session() as test_b:
#     scores = tf.random_normal([54, ], mean=1, stddev=4, seed=1)
#     boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed=1)
#     classes = tf.random_normal([54, ], mean=1, stddev=4, seed=1)
#     scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
#     print("scores[2] = " + str(scores[2].eval()))
#     print("boxes[2] = " + str(boxes[2].eval()))
#     print("classes[2] = " + str(classes[2].eval()))
#     print("scores.shape = " + str(scores.eval().shape))
#     print("boxes.shape = " + str(boxes.eval().shape))
#     print("classes.shape = " + str(classes.eval().shape))
# YOLO简单介绍
# Summary for YOLO:
# 输入维度 608 * 608 * 3
# - Input image (608, 608, 3)
# 输入图像经过CNN, 得到一个(19, 19, 5, 85)维度的输出
# - The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output.
# 之后拉伸最后两个维度变成(19, 19, 425)
# - After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
# 在19 * 19的网格中每个单元格给425个数字
# - Each cell in a 19x19 grid over the input image gives 425 numbers.
# 因为每个单元格包含5个boxes， 对应5个锚箱
# - 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture.
# 5是因为有5个数字, 80为我们要检测的数量
# - 85 = 5 + 80 where 5 is because has 5 numbers, and and 80 is the number of classes we’d like to detect
# 然后只根据几个框进行选择
# - You then select only few boxes based on:
# 根据阈值选择框，删除低于阈值的框
# - Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
# 避免重叠的框: 计算交并比
# - Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
# 以上就是YOLO最后的输出
# - This gives you YOLO’s final output.

sess = K.get_session()
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)
yolo_model = load_model("model_data/yolo.h5")
yolo_model.summary()

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)


def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

    Arguments:
        sess -- your tensorflow/Keras session containing the YOLO graph
        image_file -- name of an image stored in the "images" folder.

    Returns:
        out_scores -- tensor of shape (None, ), scores of the predicted boxes
        out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
        out_classes -- tensor of shape (None, ), class index of the predicted boxes

    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes.
    """
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    colors = generate_colors(class_names)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image.save(os.path.join("out", image_file), quality=90)
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    return out_scores, out_boxes, out_classes


# What you should remember:
# YOLO是一个牛逼的对象检测模型, 总共有五千多万个参数
# - YOLO is a state-of-the-art object detection model that is fast and accurate
# CNN读入一个图片输出19x19x5x85
# - It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume.
# 可以看成一个19 * 19的网格, 每个网格有5个boxs
# - The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
# 使用非最大抵制算法来过滤网格
# - You filter through all the boxes using non-max suppression. Specifically:
# 对所选的框只保留大于阈值的框
# - Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
# 使用交集与并集的比值(IoU)消除重叠的框
# - Intersection over Union (IoU) thresholding to eliminate overlapping boxes
# 可以用自己的数据集对YOLO进行微调, 或者直接使用就行了.
# - Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation, we used previously trained model parameters in this exercise. If you wish, you can also try fine-tuning the YOLO model with your own dataset, though this would be a fairly non-trivial exercise.