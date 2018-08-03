import input
import tensorflow as tf
import input


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


Conv1 = conv2d(X, weight_variable([4, 4, 3, 64]), bias_variable([64]))
Relu1 = relu(Conv1)
Pool1 = max_pool_2x2(Relu1)
# 32x32


Conv2 = conv2d(Pool1, weight_variable([4, 4, 64, 128]), bias_variable([128]))
Relu2 = relu(Conv2)
Pool2 = max_pool_2x2(Relu2)
# 16x16


Conv3 = conv2d(Pool2, weight_variable([4, 4, 128, 256]), bias_variable([256]))
Relu3 = relu(Conv3)
Pool3 = max_pool_2x2(Relu3)
# 8x8


Conv4 = conv2d(Pool3, weight_variable([4, 4, 256, 512]), bias_variable([512]))
Relu4 = relu(Conv4)
Pool4 = max_pool_2x2(Relu4)
# 4x4


W1 = tf.Variable(tf.truncated_normal(shape=[512*4*4, 3]))
b1 = tf.Variable(tf.truncated_normal(shape=[3]))
Pool4_flat = tf.reshape(Pool4, [-1, 512*4*4])
OutputLayer = tf.matmul(Pool4_flat, W1) + b1

Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_Label, logits=OutputLayer))
train_step = tf.train.AdamOptimizer(0.005).minimize(Loss)

correct_prediction = tf.equal(tf.arg_max(OutputLayer, 1), tf.arg_max(Y_Label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # save_path = "./model/model_1.ckpt"
    # saver.restore(sess, save_path)

    for step in range(10000):
        train_images, train_labels = input.input('train', 30)
        sess.run(train_step, feed_dict={X: train_images, Y_Label: train_labels})
        if step % 10 == 0:
            eval_images, eval_labels = input.input('eval', 20)
            print(step, sess.run(accuracy, feed_dict={X: eval_images, Y_Label: eval_labels}))

        if step % 100 == 0:
            saver = tf.train.Saver()
            save_path = saver.save(sess, "./model/model_" + str(step) + ".ckpt")
