import tensorflow as tf
from math import exp
import functools
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


def run():
    # Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
    mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

    # Learning Rate Values
    lrmax = 0.003
    lrmin = 0.00001
    decay_speed = 2000.0
    stddev = 0.1

    w = 28
    h = 28

    connect_nodes = 200
    out_nodes = 10

    # Place holders
    X = tf.placeholder(tf.float32, [None, w, h, 1])
    Y_ = tf.placeholder(tf.float32, [None, out_nodes])
    L = tf.placeholder(tf.float32)
    # pkeep = tf.placeholder(tf.float32)
    #
    # layers = [
    #     28 * 28,
    #     200,
    #     100,
    #     60,
    #     30,
    #     10
    # ]
    #
    # WW = [
    #     tf.Variable(tf.truncated_normal(
    #         [layers[i], layers[i+1]],
    #         stddev=0.1,
    #         name="Weight" + str(i)
    #     ))
    #     for i in range(len(layers)-1)
    # ]
    #
    # BB = [
    #     tf.Variable(tf.ones([layers[i]])/10, "Bias" + str(i))
    #     for i in range(1, len(layers))
    # ]

    filters = [
        [5, 5],
        [4, 4],
        [4, 4],
    ]

    strides = [
        1,
        2,
        2
    ]

    channels = [
        1,
        4,
        8,
        12
    ]

    img_reduce = functools.reduce((lambda x, y: x * y), strides)
    conv_nodes = int((w / img_reduce) * (h / img_reduce) * (channels[-1]))

    WW = [
        tf.Variable(tf.truncated_normal(filters[i] + channels[i:i+2], stddev=stddev))
        for i in range(len(filters))
    ]

    WConnect = tf.Variable(tf.truncated_normal([conv_nodes, connect_nodes], stddev=stddev))
    WOutput = tf.Variable(tf.truncated_normal([connect_nodes, out_nodes], stddev=stddev))

    BB = [
        tf.Variable(tf.truncated_normal([channels[i]], stddev=stddev))
        for i in range(1, len(channels))
    ]

    BConnect = tf.Variable(tf.ones([connect_nodes]) / 10)
    BOutput = tf.Variable(tf.ones([out_nodes]) / 10)


    # model

    Y = X

    for i in range(len(strides)):
        conv_layer = tf.nn.conv2d(Y, WW[i], strides=[1, strides[i], strides[i], 1], padding='SAME')
        Y = tf.nn.relu(conv_layer + BB[i])

    YY = tf.reshape(Y, [-1, conv_nodes])
    YConnect = tf.nn.relu(tf.matmul(YY, WConnect) + BConnect)

    Ylogits = tf.matmul(YConnect, WOutput) + BOutput
    Y = tf.nn.softmax(Ylogits)

    # loss function
    logits = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(logits) * 100

    # % of correct answers found in batch
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # optimizer = tf.train.GradientDescentOptimizer(0.003)
    optimizer = tf.train.AdamOptimizer(L)
    train_step = optimizer.minimize(cross_entropy)

    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        # load batch of images and correct answers
        batch_X, batch_Y = mnist.train.next_batch(100)
        learning_rate = lrmin + (lrmax - lrmin) * exp(-i / decay_speed)
        train_data = {
            X: batch_X,
            Y_: batch_Y,
            L: learning_rate,
        }

        # train
        sess.run(train_step, feed_dict=train_data)

    test_data = {
        X: mnist.test.images,
        Y_: mnist.test.labels,
    }
    a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
    print(a)

if __name__ == "__main__":
    run()