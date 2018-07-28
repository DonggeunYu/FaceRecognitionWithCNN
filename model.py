import input
import tensorflow as tf
import six.moves import urllib

def weight_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
    return tf.Variable(initial)

def conv2d(x, w, bias):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + bias


def relu(x):
    return tf.nn.relu(x)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
Y_Label = tf.placeholder(tf.float32, shape=[None, 3])


Conv1 = conv2d(X, weight_variable([8, 8, 3, 16]), bias_variable([16]))
Relu1 = relu(Conv1)
Pool1 = max_pool_2x2(Relu1)
# 32x32


Conv2 = conv2d(Pool1, weight_variable([4, 4, 16, 32]), bias_variable([32]))
Relu2 = relu(Conv2)
Pool2 = max_pool_2x2(Relu2)
# 16x16


Conv3 = conv2d(Pool2, weight_variable([4, 4, 32, 128]), bias_variable([128]))
Relu3 = relu(Conv3)
Pool3 = max_pool_2x2(Relu3)
# 8x8


Conv4 = conv2d(Pool3, weight_variable([4, 4, 128, 256]), bias_variable([256]))
Relu4 = relu(Conv4)
Pool4 = max_pool_2x2(Relu4)
# 4x4


Conv5 = conv2d(Pool4, weight_variable([2, 2, 256, 512]), bias_variable([512]))
Relu5 = relu(Conv5)
Pool5 = max_pool_2x2(Relu5)
#2x2


W1 = tf.Variable(tf.truncated_normal(shape=[512*2*2, 3]))
b1 = tf.Variable(tf.truncated_normal(shape=[3]))
Pool5_flat = tf.reshape(Pool5, [-1, 512*2*2])
OutputLayer = tf.matmul(Pool5_flat, W1) + b1

Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_Label, logits=OutputLayer))
train_step = tf.train.AdamOptimizer(0.001).minimize(Loss)

correct_prediction = tf.equal(tf.arg_max(OutputLayer, 1), tf.arg_max(Y_Label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        train_images, train_labels = input.get_data('train', 10)
        print(train_images[0])
        sess.run(train_step, feed_dict={X: train_images, Y_Label: train_labels})
        if i % 100 == 0:
            eval_images, eval_labels = input.get_data('eval', 10)
            print(sess.run(accuracy, feed_dict={X: eval_images, Y_Label: eval_labels}))