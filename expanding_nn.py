import itertools
import time
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
learning_rate = 1e-3

def psh(arrays):
    for array in arrays: print(np.shape(array))

def normalize(array):
    return (array - array.mean()) / array.std()
'''
Next steps:
* accuracy function
* get a model that has ~70-80 acc in mnist
* batches
'''

def run():
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [1, 784])
    y = tf.placeholder(tf.int32, [1,])
    w1 = tf.Variable(tf.random_normal((784, 10)))
    b1 = tf.Variable(tf.zeros((10)))
    scores = tf.matmul(x, w1) + b1
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
    loss = tf.reduce_mean(losses)
    params = [w1, b1]
    grad_params = tf.gradients(loss, params)
    new_ws = [tf.assign_sub(param, learning_rate * grad)
              for param, grad in zip(params, grad_params)]

    with tf.control_dependencies(new_ws):
        loss = tf.identity(loss)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        acc_loss = []
        for j in itertools.count():
            for i, (x_np, y_np) in enumerate(zip(x_train, y_train)):
                x_np = normalize(x_np.reshape(1, 784))
                y_np = y_np.reshape(1,)
                acc_loss.append(sess.run(loss, feed_dict={x: x_np, y: y_np}))

run()
