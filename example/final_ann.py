from math import exp

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


def run():
    """
    Multilayer Perceptron (5 layers)
    Dropp-off (90% change of keeping a node)
    Dynamic learning rate that reduces as time goes on. (from .003 to 0.00001)
    Activation function: relu
    Optimizer: AdamOptimizer
    :return:
    """

    # Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
    mnist = mnist_data.read_data_sets(
        "data",
        one_hot=True,
        reshape=False,
        validation_size=0
    )

    # ------ Constants -------

    # Image Format
    width = 28
    height = 28
    area = width * height

    #Data
    batch_total = 1000

    # Learning Rate Values
    lrmax = 0.003
    lrmin = 0.00001
    decay_speed = 2000.0

    # Layers
    layers = [area, 200, 100, 60, 30, 10]

    # Drop-off
    keep_ratio = 0.9

    # Epoch
    epoch_total = 5

    # ------- Placeholders -------
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y_ = tf.placeholder(tf.float32, [None, 10])
    L = tf.placeholder(tf.float32)
    pkeep = tf.placeholder(tf.float32)

    # ----- Weights and Bias -----
    WW = [
        tf.Variable(tf.truncated_normal(
            [layers[i], layers[i + 1]],
            stddev=0.1,
            name="Weight_" + str(i)
        ))
        for i in range(len(layers) - 1)
        ]

    BB = [
        tf.Variable(
            tf.ones([layers[i]]) / 10,
            name="Bias_" + str(i)
        )
        for i in range(1, len(layers))
        ]

    # ----------- Model -----------

    # Flatten image
    Y = tf.reshape(X, [-1, area])

    # ------- Activation Function -------
    i = 0
    for i in range(len(layers) - 2):
        name = "activate_" + str(i)
        Y = tf.nn.relu(tf.matmul(Y, WW[i], name=name) + BB[i])
        Y = tf.nn.dropout(Y, pkeep)

    # ------- Regression Functions -------
    Ylogits = tf.matmul(Y, WW[i + 1]) + BB[i + 1]
    Y = tf.nn.softmax(Ylogits)

    # ------- Loss Function -------
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    loss = tf.reduce_mean(cross_entropy) * 100

    # ------- Optimizer -------
    optimizer = tf.train.AdamOptimizer(L)
    train_step = optimizer.minimize(loss)

    # ------- Accuracy -------
    is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    # ------- Tensor Graph -------
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    # ------- Training -------
    for epoch in range(epoch_total):
        avg_cost = 0.

        for i in range(batch_total):
            batch_X, batch_Y = mnist.train.next_batch(100)
            learning_rate = lrmin + (lrmax - lrmin) * exp(-i / decay_speed)
            train_data = {
                X: batch_X,
                Y_: batch_Y,
                L: learning_rate,
                pkeep: keep_ratio
            }

            _, c = sess.run(
                [train_step, loss], # Operations to run
                feed_dict=train_data
            )

            avg_cost += c / batch_total

        # Display logs per epoch step
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    # ------- Testing -------
    test_data = {X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0}
    a, c = sess.run([accuracy, loss], feed_dict=test_data)
    print(a)
