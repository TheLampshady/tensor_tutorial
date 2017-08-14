#!/usr/bin/env python3
from os.path import basename, splitext

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

from tensor_functions import variable_summaries


def run():
    # Download images and labels into mnist.test (10K images+labels) and
    #   mnist.train (60K images+labels)
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
    output = 10

    lr = 0.003

    hidden_layer = 200

    batch_total = 1000

    # Tensor Board Log
    logs_path = "tensor_log/" + splitext(basename(__file__))[0]

    # ------- Placeholders -------
    X = tf.placeholder(tf.float32, [None, width, height, 1], name="Input_PH")
    Y_ = tf.placeholder(tf.float32, [None, 10], name="Output_PH")

    XX = tf.reshape(X, [-1, area])

    # ----- Weights and Bias -----
    # - Added Hidden Layer (W2 and B2)

    with tf.name_scope('Layer'):
        with tf.name_scope('Weights'):
            weights1 = tf.Variable(
                tf.truncated_normal([area, hidden_layer], stddev=0.1, name="Weights_Init"),
                name="Weights"
            )
            variable_summaries(weights1, "Weights")
        with tf.name_scope('Biases'):
            biases1 = tf.Variable(tf.zeros([hidden_layer]), name="Biases")
            variable_summaries(biases1, "Biases")
    with tf.name_scope('Layer'):
        with tf.name_scope('Weights'):
            weights2 = tf.Variable(
                tf.truncated_normal([hidden_layer, output], stddev=0.1, name="Weights_Init"),
                name="Weights"
            )
            variable_summaries(weights2, "Weights")
        with tf.name_scope('Biases'):
            biases2 = tf.Variable(tf.zeros([output]), name="Biases")
            variable_summaries(biases2, "Biases")

    # ------- Activation Function -------
    # - Used for Hidden (2nd) Layer
    Y1 = tf.nn.sigmoid(
        tf.matmul(XX, weights1, name="Product") + biases1
    )

    # ------- Regression Functions -------
    Y = tf.nn.softmax(
        tf.matmul(Y1, weights2, name="Product") + biases2,
        name="Output_Result"
    )

    # ------- Loss Function -------
    loss = -tf.reduce_sum(Y_ * tf.log(Y), name="Loss")
    tf.summary.scalar('Losses', loss)

    # ------- Optimizer -------
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_step = optimizer.minimize(loss)

    # ------- Accuracy -------
    is_correct = tf.equal(
        tf.argmax(Y, 1, name="Max_Output"),
        tf.argmax(Y_, 1, name="Max_Target")
    )
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    tf.summary.scalar('Accuracies', accuracy)

    # ------- Tensor Graph -------
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Tensor Board
    tensor_graph = tf.get_default_graph()
    summary_writer = tf.summary.FileWriter(logs_path, graph=tensor_graph)

    # ------- Training -------
    for i in range(batch_total):
        # load batch of images and correct answers
        batch_X, batch_Y = mnist.train.next_batch(100)
        train_data = {X: batch_X, Y_: batch_Y}

        # train
        _, c = sess.run([train_step, loss], feed_dict=train_data)

    test_data = {X: mnist.test.images, Y_: mnist.test.labels}
    a, c = sess.run([accuracy, loss], feed_dict=test_data)
    print(a)

if __name__ == "__main__":
    run()
