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

    # Data
    batch_total = 1000
    batch_size = 100
    test_freq = 10

    # Tensor Board Log
    logs_path = "tensor_log/" + splitext(basename(__file__))[0]

    # ------- Placeholders -------
    X = tf.placeholder(tf.float32, [None, width, height, 1], name="Input_PH")
    Y_ = tf.placeholder(tf.float32, [None, 10], name="Output_PH")

    input_flat = tf.reshape(X, [-1, area])

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
    with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_flat, weights1, name="Product") + biases1
        tf.summary.histogram('Pre_Activations', preactivate)
        activations = tf.nn.sigmoid(preactivate)
        tf.summary.histogram('Activations', activations)

    # ------- Regression Functions -------
    with tf.name_scope('Wx_plus_b'):
        logits = tf.matmul(activations, weights2, name="Product") + biases2
        tf.summary.histogram('Pre_Activations', logits)
    Y = tf.nn.softmax(logits, name="Output_Result")

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
    # - Exports the model to be used by Tensor Board
    merged_summary_op = tf.summary.merge_all()

    tensor_graph = tf.get_default_graph()
    train_writer = tf.summary.FileWriter(logs_path + "/train", graph=tensor_graph)
    test_writer = tf.summary.FileWriter(logs_path + "/test")

    # ------- Training -------
    avg_cost = 0.
    train_operations = [train_step, loss, merged_summary_op]
    test_operations = [accuracy, loss, merged_summary_op]
    test_data = {X: mnist.test.images, Y_: mnist.test.labels}

    for step in range(batch_total):
        # load batch of images and correct answers
        batch_X, batch_Y = mnist.train.next_batch(batch_size)
        train_data = {X: batch_X, Y_: batch_Y}

        # train
        _, cross_loss, summary = sess.run(
            train_operations,
            feed_dict=train_data
        )

        avg_cost += cross_loss / batch_total
        train_writer.add_summary(summary, step)

        if step % test_freq == 0:
            acc, cross_loss, summary = sess.run(
                test_operations,
                feed_dict=test_data
            )
            test_writer.add_summary(summary, step)
            print('Accuracy at step %s: %s' % (step, acc))

    print("Cost: ", "{:.9f}".format(avg_cost))

if __name__ == "__main__":
    run()
