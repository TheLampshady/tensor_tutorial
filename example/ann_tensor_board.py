from math import exp
from os.path import basename, splitext

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


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

    # Data
    batch_total = 1000
    batch_size = 100

    # Learning Rate Values
    lrmax = 0.003
    lrmin = 0.00001
    decay_speed = 2000.0

    # Layers
    layers = [area, 200, 100, 60, 30, 10]

    # Drop-off
    keep_ratio = 0.9

    # Epoch
    epoch_total = 1

    # Tensor Board Log
    logs_path = "tensor_log/" + splitext(basename(__file__))[0]

    # ------- Placeholders -------
    X = tf.placeholder(tf.float32, [None, width, height, 1], name="Input")
    Y_ = tf.placeholder(tf.float32, [None, 10], name="Output")
    L = tf.placeholder(tf.float32, name="Learning_Rate")
    pkeep = tf.placeholder(tf.float32, name="Percentage_Keep")

    # ----- Weights and Bias -----
    WW = [
        tf.Variable(tf.truncated_normal(
            [layers[i], layers[i + 1]],
            stddev=0.1,
            name="Init_Weights"
        ), name="Weights" + str(i))
        for i in range(len(layers) - 1)
        ]

    BB = [
        tf.Variable(
            tf.ones([layers[i]]) / 10,
            name="Bias_" + str(i-1)
        )
        for i in range(1, len(layers))
    ]

    # ---------------- Operations ----------------

    # Flatten image
    Y = tf.reshape(X, [-1, area])

    # ------- Activation Function -------
    i = 0
    for i in range(len(layers) - 2):
        result = tf.add(tf.matmul(Y, WW[i], name="Product"), BB[i], name="Plus")
        Y = tf.nn.relu(result)
        Y = tf.nn.dropout(Y, pkeep, name="Dropout")

    # ------- Regression Functions -------
    i += 1
    logits = tf.add(tf.matmul(Y, WW[i], name="Product"), BB[i], name="Plus")
    Y = tf.nn.softmax(logits, name="final_result")

    # ---------------- Operations ----------------

    # ------- Loss Function -------
    # with tf.name_scope('Loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y_, name="cross_entropy")
    loss = tf.reduce_mean(cross_entropy, name="loss") * 100

    # ------- Optimizer -------
    # with tf.name_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer(L, name="adam")
    train_step = optimizer.minimize(loss, name="minimize")

    # ------- Accuracy -------
    # with tf.name_scope('Accuracy'):
    is_correct = tf.equal(tf.argmax(Y, 1, name="y_arg_max"), tf.argmax(Y_, 1, name="target"))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    # ------- Tensor Graph -------
    # Start Tensor Graph
    init = tf.global_variables_initializer()

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", loss)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    sess = tf.Session()
    sess.run(init)

    # Tensor Board
    tensor_graph = tf.get_default_graph()
    summary_writer = tf.summary.FileWriter(logs_path, graph=tensor_graph)

    # ------- Training -------
    for epoch in range(epoch_total):
        avg_cost = 0.

        for i in range(batch_total):
            batch_X, batch_Y = mnist.train.next_batch(batch_size)
            learning_rate = lrmin + (lrmax - lrmin) * exp(-i / decay_speed)
            train_data = {
                X: batch_X,
                Y_: batch_Y,
                L: learning_rate,
                pkeep: keep_ratio
            }

            _, c, summary = sess.run(
                [train_step, loss, merged_summary_op], # Operations to run
                feed_dict=train_data
            )

            avg_cost += c / batch_total

            summary_writer.add_summary(summary, epoch * batch_total + i)

        # Display logs per epoch step
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    # ------- Testing -------
    test_data = {X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0}
    a, c = sess.run([accuracy, loss], feed_dict=test_data)
    print(a)
