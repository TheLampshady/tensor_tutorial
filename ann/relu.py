#!/usr/bin/env python3
from os.path import basename, splitext

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

from tensor_functions import bias_variable, weight_variable


def run():
    """
    Example 3
    Multilayer Perceptron (5 layers)
    Activation function: relu
    Optimizer: AdamOptimizer
    :return:
    """

    mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

    # Image Format
    width = 28
    height = 28
    area = width * height
    output = 10
    channels = 1

    # Data
    batch_total = 1001
    batch_size = 100
    test_freq = 10

    # Learning Rate
    lr = 0.003

    # Layers
    layers = [
        area,
        200,
        100,
        60,
        30,
        output
    ]

    # Tensor Board Log
    logs_path = "tensor_log/%s/" % splitext(basename(__file__))[0]

    # ------- Placeholders -------
    X = tf.placeholder(tf.float32, [None, width, height, channels], name="Input_PH")
    Y_ = tf.placeholder(tf.float32, [None, output], name="Output_PH")

    Y = tf.reshape(X, [-1, area])

    # ----- Weights and Bias -----
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        with tf.name_scope('Layer'):
            weights.append(weight_variable([layers[i], layers[i + 1]]))
            biases.append(bias_variable([layers[i + 1]]))

    # ---------------- Operations ----------------

    # ------- Activation Function -------
    i = 0
    for i in range(len(layers)-2):
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(Y, weights[i], name="Product") + biases[i]
            tf.summary.histogram('Pre_Activations', preactivate)
            # Y = tf.nn.sigmoid(preactivate)
            Y = tf.nn.relu(preactivate)
            tf.summary.histogram('Activations', Y)

    # ------- Regression Functions -------
    i += 1
    with tf.name_scope('Wx_plus_b'):
        logits = tf.matmul(Y, weights[i], name="Product") + biases[i]
        tf.summary.histogram('Pre_Activations', logits)
    Y = tf.nn.softmax(logits, name="Output_Result")

    # ------- Loss Function -------
    with tf.name_scope('Loss'):
        # cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y_, name="Cross_Entropy")
        with tf.name_scope('Total'):
            loss = tf.reduce_mean(cross_entropy, name="loss") * 100
    tf.summary.scalar('Losses', loss)

    # ------- Optimizer -------
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(lr)
        train_step = optimizer.minimize(loss, name="minimize")

    # ------- Accuracy -------
    with tf.name_scope('Accuracy'):
        with tf.name_scope('correct_prediction'):
            is_correct = tf.equal(tf.argmax(Y, 1, name="Max_Result"), tf.argmax(Y_, 1, name="Target"))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    tf.summary.scalar('Accuracies', accuracy)

    # ------- Tensor Graph -------
    # Start Tensor Graph
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    # Tensor Board
    merged_summary_op = tf.summary.merge_all()

    tensor_graph = tf.get_default_graph()
    train_writer = tf.summary.FileWriter(logs_path + "train", graph=tensor_graph)
    test_writer = tf.summary.FileWriter(logs_path + "test")

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
