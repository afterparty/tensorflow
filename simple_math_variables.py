import tensorflow as tf

#y = Wx + b
w = tf.Variable([2.5, 4.0], tf.float32, name='var_W')

x = tf.placeholder(tf.float32, name='x')
b = tf.Variable([5.0, 10.0], tf.float32, name='var_b')

y = w * x + b

# init variables (must do to get variables)
init = tf.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    print("Final result: Wx + b = ", sess.run(y, feed_dict={x: [10, 100]}))

s = w * x

init = tf.compat.v1.variables_initializer([w])

with tf.compat.v1.Session() as sess:
    sess.run(init)

    print("Result: Wx = ", sess.run(s, feed_dict={x: [10, 100]}))

number = tf.Variable(2)
multipler = tf.Variable(1)

init = tf.global_variables_initializer()

result = number.assign(tf.multiply(number, multipler))

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for i in range(10):
        print("Result number * multiplier = ", sess.run(result))
        print("Increment multiplier, new value = ",
              sess.run(multipler.assign_add(1)))
