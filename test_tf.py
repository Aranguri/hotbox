import tensorflow as tf
import numpy as np

tf.reset_default_graph()

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
loss = x * y + x
grads = tf.gradients(loss, x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_np = [2]
    y_np = [3]
    to_print = sess.run(grads, feed_dict={x: x_np, y: y_np})
    print(to_print)
