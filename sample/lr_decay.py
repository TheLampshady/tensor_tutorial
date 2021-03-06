#!/usr/bin/env python3
from math import exp
from os.path import basename, splitext, join

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

from tensor_functions import bias_variable, weight_variable, build_mnist_embeddings, embedding_initializer


def run():
    """
    Example 4
    Multilayer Perceptron (5 layers)
    Dynamic learning rate that reduces as time goes on. (from .003 to 0.00001)
    Activation function: relu
    Optimizer: AdamOptimizer
    :return:
    """

    mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
    sprite_path, label_path = build_mnist_embeddings('data', mnist)

    # Image Format
    width = 28
    height = 28
    area = width * height
    output = 10
    channels = 1

    # Data
    epoch_total = 3

    batch_total = 1001
    batch_size = 100
    test_freq = 10 * epoch_total

    # Learning Rate Values
    lrmax = 0.003
    lrmin = 0.00001
    decay_speed = 2000.0

    get_lr = lambda x: lrmin + (lrmax - lrmin) * exp(-x / decay_speed)

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
    embed_path = join(logs_path, "model.ckpt")
    embedding_size = 1024

    # ------- Placeholders -------
    X = tf.placeholder(tf.float32, [None, width, height, channels], name="Input_PH")
    Y_ = tf.placeholder(tf.float32, [None, output], name="Output_PH")
    L = tf.placeholder(tf.float32, name="Learning_Rate_PH")

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
    activations = None
    layer_count = range(len(layers)-2)
    for i in layer_count:
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(Y, weights[i], name="Product") + biases[i]
            tf.summary.histogram('Pre_Activations', preactivate)
            if i == layer_count[-1]:
                activations = Y = tf.nn.relu(preactivate)
            else:
                Y = tf.nn.relu(preactivate)
            tf.summary.histogram('Activations', Y)

    # ------- Regression Functions -------
    i += 1
    with tf.name_scope('Wx_plus_b'):
        logits = tf.matmul(activations, weights[i], name="Product") + biases[i]
        tf.summary.histogram('Pre_Activations', logits)
        prediction = tf.nn.softmax(logits, name="Output_Result")

    # ------- Loss Function -------
    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y_, name="Cross_Entropy")
        with tf.name_scope('Total'):
            loss = tf.reduce_mean(cross_entropy, name="loss") * 100
    tf.summary.scalar('Losses', loss)

    # ------- Optimizer -------
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(L)
        train_step = optimizer.minimize(loss, name="minimize")

    # ------- Accuracy -------
    with tf.name_scope('Accuracy'):
        with tf.name_scope('correct_prediction'):
            is_correct = tf.equal(tf.argmax(prediction, 1, name="Max_Result"), tf.argmax(Y_, 1, name="Target"))
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

    # Embeddings
    assignment = embedding_initializer(
        activations,
        embedding_size,
        test_writer,
        [height, width],
        sprite_path,
        label_path,
    )
    saver = tf.train.Saver()

    # ------- Training -------
    avg_cost = 0.
    train_operations = [train_step, loss, merged_summary_op]
    test_operations = [accuracy, loss, merged_summary_op]
    test_data = {X: mnist.test.images, Y_: mnist.test.labels}
    embed_data = {X: mnist.test.images[:embedding_size], Y_: mnist.test.labels[:embedding_size]}

    for epoch in range(epoch_total):
        avg_cost = 0.
        for i in range(batch_total):
            step = (batch_total * epoch) + i

            # ----- Train step -----
            batch_X, batch_Y = mnist.train.next_batch(batch_size)
            learning_rate = get_lr(step)
            train_data = {X: batch_X, Y_: batch_Y, L: learning_rate}

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

            # ----- Embedding -----
            if step % 500 == 0:
                sess.run(assignment, feed_dict=embed_data)
                saver.save(sess, embed_path, step)

        print("Cost: ", "{:.9f}".format(avg_cost))


if __name__ == "__main__":
    run()
