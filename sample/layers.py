#!/usr/bin/env python3
from os.path import basename, splitext, join

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

from tensor_functions import variable_summaries, embedding_initializer, build_mnist_embeddings


def run():
    """
    Example 2
    Multilayer Perceptron (2 layers)
    Activation function: sigmoid
    Optimizer: GradientDescentOptimizer
    :return:
    """

    mnist = mnist_data.read_data_sets(
        "data",
        one_hot=True,
        reshape=False,
        validation_size=0
    )
    sprite_path, label_path = build_mnist_embeddings('data', mnist)

    # ------ Constants -------

    # Image Format
    width = 28
    height = 28
    area = width * height
    output = 10

    lr = 0.003

    hidden_layer = 300

    # Data
    batch_total = 1001
    batch_size = 100
    test_freq = 10

    # Tensor Board Log
    logs_path = "tensor_log/%s/" % splitext(basename(__file__))[0]
    embed_path = join(logs_path, "model.ckpt")
    embedding_batch = 1024

    # ------- Placeholders -------
    X = tf.placeholder(tf.float32, [None, width, height, 1], name="Input_PH")
    Y_ = tf.placeholder(tf.float32, [None, 10], name="Output_PH")

    input_flat = tf.reshape(X, [-1, area])

    # ----- Weights and Bias -----
    # - Added Hidden Layer (W2 and B2)

    with tf.name_scope('Layer'):

        weights1 = tf.Variable(
            tf.truncated_normal([area, hidden_layer], stddev=0.1, name="Weights_Init"),
            name="Weights"
        )
        with tf.name_scope('Weights'):
            variable_summaries(weights1, "Weights")

        biases1 = tf.Variable(tf.ones([hidden_layer]) / 10, name="Biases")
        with tf.name_scope('Biases'):
            variable_summaries(biases1, "Biases")
    with tf.name_scope('Layer'):
        weights2 = tf.Variable(
            tf.truncated_normal([hidden_layer, output], stddev=0.1, name="Weights_Init"),
            name="Weights"
        )
        with tf.name_scope('Weights'):
            variable_summaries(weights2, "Weights")

        biases2 = tf.Variable(tf.ones([output]) / 10, name="Biases")
        with tf.name_scope('Biases'):
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
        prediction = tf.nn.softmax(logits, name="Output_Result")

    # ------- Loss Function -------
    loss = -tf.reduce_sum(Y_ * tf.log(prediction), name="Loss")
    tf.summary.scalar('Losses', loss)

    # ------- Optimizer -------
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_step = optimizer.minimize(loss)

    # ------- Accuracy -------
    is_correct = tf.equal(
        tf.argmax(prediction, 1, name="Max_Output"),
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
    train_writer = tf.summary.FileWriter(logs_path + "train", graph=tensor_graph)
    test_writer = tf.summary.FileWriter(logs_path + "test")

    # Embeddings
    assignment = embedding_initializer(
        activations,
        embedding_batch,
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
    embed_data = {X: mnist.test.images[:embedding_batch], Y_: mnist.test.labels[:embedding_batch]}

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

        # ----- Embedding -----
        if step % 500 == 0:
            sess.run(assignment, feed_dict=embed_data)
            saver.save(sess, embed_path, step)

    print("Cost: ", "{:.9f}".format(avg_cost))

if __name__ == "__main__":
    run()
