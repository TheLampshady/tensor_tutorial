#!/usr/bin/env python3
from os.path import basename, splitext

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

from tensor_functions import variable_summaries


def run():
    # Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
    mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

    # ------ Constants -------

    # Image Format
    width = 28
    height = 28
    area = width * height
    output = 10

    lr = 0.003

    # Data
    batch_total = 1001
    batch_size = 100
    test_freq = 10

    # Tensor Board Log
    logs_path = "tensor_log/" + splitext(basename(__file__))[0]

    # ------- Placeholders -------
    X = tf.placeholder(tf.float32, [None, width, height, 1], name="Input_PH")
    Y_ = tf.placeholder(tf.float32, [None, output], name="Output_PH")

    input_flat = tf.reshape(X, [-1, area])

    # ----- Weights and Bias -----
    with tf.name_scope('Layer'):
        with tf.name_scope('Weights'):
            weights = tf.Variable(tf.zeros([area, 10], name="Weights_Init"), name="Weights")
            variable_summaries(weights, "Weights")
        with tf.name_scope('Biases'):
            biases = tf.Variable(tf.zeros([10], name="Biases_Init"), name="Biases")
            variable_summaries(biases, "Biases")

    # ------- Regression Function -------
    with tf.name_scope('Wx_plus_b'):
        logits = tf.matmul(input_flat, weights, name="Product") + biases
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
    # init = tf.initialize_all_variables()
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
