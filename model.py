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


X = tf.placeholder(tf.float32, [None, FLAGS.images_size, FLAGS.images_size, FLAGS.image_color])
Y_Label = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for in
# 이미지 불러오기
#img = cv2.imread('Face_Image/' + '문재인/')