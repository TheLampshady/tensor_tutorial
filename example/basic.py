from os.path import basename, splitext

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


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

    # Tensor Board Log
    logs_path = "tensor_log/" + splitext(basename(__file__))[0]

    # ------- Placeholders -------
    X = tf.placeholder(tf.float32, [None, width, height, 1], name="Input_PH")
    Y_ = tf.placeholder(tf.float32, [None, output], name="Output_PH")

    # ----- Weights and Bias -----
    W = tf.Variable(tf.zeros([area, 10], name="Weight_Init"), name="Weight")
    b = tf.Variable(tf.zeros([10], name="Bias_Init"), name="Bias")

    XX = tf.reshape(X, [-1, area])

    # ------- Activation Function -------
    Y = tf.nn.softmax(
        tf.add(tf.matmul(XX, W), b)
    )

    # ------- Loss Function -------
    loss = -tf.reduce_sum(Y_ * tf.log(Y), name="Loss")

    # ------- Optimizer -------
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_step = optimizer.minimize(loss)

    # ------- Accuracy -------
    is_correct = tf.equal(
        tf.argmax(Y, 1, name="Max_Output"),
        tf.argmax(Y_, 1, name="Target")
    )
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    # ------- Tensor Graph -------
    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    # Tensor Board
    tensor_graph = tf.get_default_graph()
    tf.summary.FileWriter(logs_path, graph=tensor_graph)

    for i in range(1000):
        # load batch of images and correct answers
        batch_X, batch_Y = mnist.train.next_batch(100)
        train_data = {X: batch_X, Y_: batch_Y}

        # train
        sess.run(train_step, feed_dict=train_data)

    test_data = {X: mnist.test.images, Y_: mnist.test.labels}
    a, c = sess.run([accuracy, loss], feed_dict=test_data)
    print(a)

if __name__ == "__main__":
    run()