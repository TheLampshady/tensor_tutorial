#!/usr/bin/env python3
from math import exp
from os.path import basename, splitext

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

from tensor_functions import bias_variable, weight_variable


def run():
    """
    Multilayer Perceptron (5 layers)
    Drop-off (90% change of keeping a node)
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
    output = 10
    channels = 1

    # Data
    batch_total = 1000
    batch_size = 100

    # Learning Rate Values
    lrmax = 0.003
    lrmin = 0.00001
    decay_speed = 2000.0

    # Layers
    layers = [area, 200, 100, 60, 30, output]

    # Drop-off
    keep_ratio = 0.9

    # Epoch
    epoch_total = 1

    # Tensor Board Log
    logs_path = "tensor_log/" + splitext(basename(__file__))[0]

    # ------- Placeholders -------
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, width, height, channels], name="Input_PH")
        Y_ = tf.placeholder(tf.float32, [None, output], name="Output_PH")

    L = tf.placeholder(tf.float32, name="Learning_Rate_PH")
    keep_prob = tf.placeholder(tf.float32, name="Per_Keep_PH")

    with tf.name_scope('input_reshape'):
        Y = tf.reshape(X, [-1, area])

    # ----- Weights and Bias -----
    weights = [
        weight_variable([layers[i], layers[i + 1]])
        for i in range(len(layers) - 1)
        ]

    biases = [
        bias_variable([layers[i]])
        for i in range(1, len(layers))
    ]

    # ---------------- Operations ----------------

    # ------- Activation Function -------
    i = 0
    for i in range(len(layers) - 2):
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.add(tf.matmul(Y, weights[i], name="Product"), biases[i], name="Plus")
            tf.summary.histogram('pre_activations', preactivate)
            activations = tf.nn.relu(preactivate)
            tf.summary.histogram('activations', activations)

        with tf.name_scope('dropout'):
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            Y = tf.nn.dropout(activations, keep_prob, name="Dropout")

    # ------- Regression Functions -------
    i += 1
    logits = tf.add(tf.matmul(Y, weights[i], name="Product"), biases[i], name="Plus")
    Y = tf.nn.softmax(logits, name="Final_Result")

    # ---------------- Operations ----------------

    # ------- Loss Function -------
    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y_, name="cross_entropy")
        with tf.name_scope('Total'):
            loss = tf.reduce_mean(cross_entropy, name="loss") * 100
    tf.summary.scalar('Loss', loss)

    # ------- Optimizer -------
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(L)
        train_step = optimizer.minimize(loss, name="minimize")

    # ------- Accuracy -------
    with tf.name_scope('Accuracy'):
        with tf.name_scope('correct_prediction'):
            is_correct = tf.equal(tf.argmax(Y, 1, name="Max_Result"), tf.argmax(Y_, 1, name="Target"))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # ------- Tensor Graph -------
    # Start Tensor Graph
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    # Tensor Board
    merged_summary_op = tf.summary.merge_all()

    tensor_graph = tf.get_default_graph()
    train_writer = tf.summary.FileWriter(logs_path + "/train", graph=tensor_graph)
    test_writer = tf.summary.FileWriter(logs_path + "/test")

    # ------- Training -------
    test_data = {X: mnist.test.images, Y_: mnist.test.labels, keep_prob: 1.0}
    for epoch in range(epoch_total):
        avg_cost = 0.

        for i in range(batch_total):
            step = (batch_total * epoch) + i

            # ----- Test Step -----
            if i % 10 == 0:
                acc, cross_loss, summary = sess.run(
                    [accuracy, loss, merged_summary_op],
                    feed_dict=test_data
                )
                test_writer.add_summary(summary, step)
                print('Accuracy at step %s: %s' % (step, acc))

            # ----- Train step -----
            batch_X, batch_Y = mnist.train.next_batch(batch_size)

            learning_rate = lrmin + (lrmax - lrmin) * exp(-step / decay_speed)
            train_data = {
                X: batch_X,
                Y_: batch_Y,
                L: learning_rate,
                keep_prob: keep_ratio
            }

            train_operations = [train_step, loss, merged_summary_op]
            # Record execution stats
            if i % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                _, cross_loss, summary = sess.run(
                    train_operations,
                    feed_dict=train_data,
                    options=run_options,
                    run_metadata=run_metadata
                )

            else:
                _, cross_loss, summary = sess.run(
                    train_operations,
                    feed_dict=train_data
                )

            avg_cost += cross_loss / batch_total
            train_writer.add_summary(summary, step)

        # Display logs per epoch step
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))


if __name__ == "__main__":
    run()
