import tensorflow as tf
import model
import input

def train():
    keep_prob = tf.placeholder(tf.float32)
    images, labels = input.get_data('train', 10)
    hypothesis, cross_entropy, train_step = model.make_network(images, labels, keep_prob)

    coss_sum = tf.summary.scalar('cast', cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./logscost_log')
        writer.add_graph(sess.graph)

        merge_sum = tf.summary.marge_all()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(1000):
            summary, _ = sess.run([merge_sum, train_step], feed_dict={keep_prob: 0.7})
            writer.add_summary(summary, global_step=step)
            print(step, sess.run(cross_entropy, feed_dict={keep_prob: 1.0}))

        coord.request_stop()
        coord.join(threads)

def main(argv = None):
    train()


if __name__ == '__name__':
    tf.app.run()