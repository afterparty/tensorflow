import tensorflow as tf
import numpy as np

sess = tf.Session()

zeroD = np.array(30, dtype=np.int32)
sess.run(tf.rank(zeroD))

oneD = np.array([5.6, 6.3, 8.9, 9.0], dtype=np.float32)
sess.run(tf.rank(oneD))
