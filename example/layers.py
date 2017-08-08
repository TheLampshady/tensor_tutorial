from os.path import basename, splitext

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


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

    # Tensor Board Log
    logs_path = "tensor_log/" + splitext(basename(__file__))[0]

    # ------- Placeholders -------
    X = tf.placeholder(tf.float32, [None, width, height, 1], name="Input_PH")
    Y_ = tf.placeholder(tf.float32, [None, 10], name="Output_PH")

    # ----- Weights and Bias -----
    # - Added Hidden Layer (W2 and B2)
    W1 = tf.Variable(
        tf.truncated_normal([area, 200], stddev=0.1),
        name="Weight_1"
    )

    W2 = tf.Variable(
        tf.truncated_normal([200, output], stddev=0.1),
        name="Weight_2"
    )

    B1 = tf.Variable(tf.zeros([200]), name="Bias_1")
    B2 = tf.Variable(tf.zeros([output]), name="Bias_2")

    XX = tf.reshape(X, [-1, area])

    # ------- Activation Function -------
    # - Used for Hidden (2nd) Layer
    Y1 = tf.nn.sigmoid(
        tf.add(tf.matmul(XX, W1), B1),
        name="Activation"
    )

    # ------- Regression Functions -------
    Y = tf.nn.softmax(
        tf.add(tf.matmul(Y1, W2), B2),
    )

    # ------- Loss Function -------
    loss = -tf.reduce_sum(Y_ * tf.log(Y), name="Loss")

    # ------- Optimizer -------
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_step = optimizer.minimize(loss)

    # ------- Accuracy -------
    is_correct = tf.equal(
        tf.argmax(Y, 1, name="max_output"),
        tf.argmax(Y_, 1, name="Target")
    )
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    # ------- Tensor Graph -------
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Tensor Board
    tensor_graph = tf.get_default_graph()
    summary_writer = tf.summary.FileWriter(logs_path, graph=tensor_graph)


    for i in range(1000):
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