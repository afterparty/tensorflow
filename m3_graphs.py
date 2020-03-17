import tensorflow as tf

config = tf.ConfigProto(
    device_count={'GPU': 0}
)

g1 = tf.Graph()

with g1.as_default():
    with tf.compat.v1.Session(config=config) as sess:

        A = tf.constant([5, 7], tf.int32, name='A')

        x = tf.compat.v1.placeholder(tf.int32, name='x')
        b = tf.constant([3, 4], tf.int32, name='b')

        y = A * x + b

        print(sess.run(y, feed_dict={x: [10, 100]}))

        assert y.graph is g1

g2 = tf.Graph()

with g2.as_default():
    with tf.compat.v1.Session(config=config) as sess:

        A = tf.constant([5, 7], tf.int32, name='A')

        x = tf.compat.v1.placeholder(tf.int32, name='x')

        y = tf.compat.v1.pow(A, x, name="y")

        print(sess.run(y, feed_dict={x: [3, 5]}))

        assert y.graph is g2

default_graph = tf.compat.v1.get_default_graph()
with tf.compat.v1.Session(config=config) as sess:
    A = tf.constant([5, 7], tf.int32, name='A')

    x = tf.compat.v1.placeholder(tf.int32, name='x')

    y = A + x
    print(sess.run(y, feed_dict={x: [3, 5]}))

    assert y.graph is default_graph
