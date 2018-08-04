import input
import tensorflow as tf
import input
import numpy as np

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def weight_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))
    return tf.Variable(initial)

def conv2d(x, w, bias):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + bias


def relu(x):
    return tf.nn.relu(x)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
Y_Label = tf.placeholder(tf.float32, shape=[None, 3])
keep_prob = tf.placeholder(tf.float32)


Conv1 = conv2d(X, weight_variable([5, 5, 3, 64]), bias_variable([64]))
Relu1 = relu(Conv1)
Pool1 = max_pool_2x2(Relu1)
# 32x32


Conv2 = conv2d(Pool1, weight_variable([5, 5, 64, 64]), bias_variable([64]))
Relu2 = relu(Conv2)
Pool2 = max_pool_2x2(Relu2)
# 16x16


Conv3 = conv2d(Pool2, weight_variable([4, 4, 64, 128]), bias_variable([128]))
Relu3 = relu(Conv3)
Pool3 = max_pool_2x2(Relu3)
# 8x8


Conv4 = conv2d(Pool3, weight_variable([4, 4, 128, 128]), bias_variable([128]))
Relu4 = relu(Conv4)
Pool4 = max_pool_2x2(Relu4)
# 4x4

w_fc1 = tf.Variable(tf.truncated_normal(shape=[4 * 4 * 128, 512], stddev=5e-2))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]))

h_conv5_flat = tf.reshape(Pool4, [ -1, 4 * 4 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, w_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = tf.Variable(tf.truncated_normal(shape=[512, 3], stddev=5e-2))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[3]))
logits = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
y_pred = tf.nn.softmax(logits)

x_train, y_train = input.input('train', 50000)
x_test, y_test = input.input('eval', 10000)

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 3), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 3), axis=1)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_Label, logits=logits))
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y_Label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # save_path = "./model/model_1.ckpt"
    # saver.restore(sess, save_path)

    for step in range(10000):
        batch = next_batch(64, x_train, y_train_one_hot.eval())

        if step % 10 == 0:

            accuracy_print = accuracy.eval(feed_dict={X: batch[0], Y_Label: batch[1], keep_prob: 1.0})
            Loss_print = loss.eval(feed_dict={X: batch[0], Y_Label: batch[1], keep_prob: 1.0})
            print(step, accuracy_print, Loss_print)

        sess.run(train_step, feed_dict={X: batch[0], Y_Label: batch[1], keep_prob: 0.8})