import tensorflow as tf
import os
from PIL import Image
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
FLAGS.images_size = 64
FLAGS.image_color = 3
FLAGS.maxpool_filter_size = 2
FLAGS.num_classes = 5
FLAGS.batch_size = 100
FLAGS.learning_rate = 0.0001
FLAGS.log_dir = 'log/'


def main():
    images = tf.placeholder(tf.float32, [None, FLAGS.images_size, FLAGS.images_size, FLAGS.image_color])
    labels = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    image_batch, label
Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=))
# 이미지 불러오기
#img = cv2.imread('Face_Image/' + '문재인/')