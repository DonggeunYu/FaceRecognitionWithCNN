import tensorflow as tf
import os
import cv2

def weight_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
    return initial


def bias_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
    return initial


def conv2d(x, w, bias):
    initial = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + bias
    return initial

def max_pool_2x2(x):
    initial = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return initial

def Reru(x):
    return tf.nn.relu(x)


print(os.listdir('Face_Image/' + '문재인/'))
# 이미지 불러오기
#img = cv2.imread('Face_Image/' + '문재인/')