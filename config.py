import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('raw_height', 64, '')
tf.app.flags.DEFINE_integer('raw_width', 64, '')
tf.app.flags.DEFINE_integer('height', 64, '')
tf.app.flags.DEFINE_integer('width', 64, '')
tf.app.flags.DEFINE_integer('depth', 3, '')

tf.app.flags.DEFINE_string('data_dir', './data', '')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint', '')
