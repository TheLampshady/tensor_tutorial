#Step 3

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


def run():
    # Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
    mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y_ = tf.placeholder(tf.float32, [None, 10])

    layers = [
        28 * 28,
        200,
        100,
        60,
        30,
        10
    ]

    WW = [
        tf.Variable(tf.truncated_normal(
            [layers[i], layers[i+1]],
            stddev=0.1,
            name="Weight" + str(i)
        ))
        for i in range(len(layers)-1)
    ]

    BB = [
        # tf.Variable(tf.zeros([layers[i]]), "Bias" + str(i))
        tf.Variable(tf.ones([layers[i]])/10, "Bias" + str(i))
        for i in range(1, len(layers))
    ]


    # model
    Y = tf.reshape(X, [-1, 28 * 28])

    i = 0
    for i in range(len(layers)-2):
        name = "activate_" + str(i)
        # Y = tf.nn.sigmoid(tf.matmul(Y, WW[i], name=name) + BB[i])
        Y = tf.nn.relu(tf.matmul(Y, WW[i], name=name) + BB[i])

    Ylogits = tf.matmul(Y, WW[i+1]) + BB[i+1]
    Y = tf.nn.softmax(Ylogits)

    # loss function
    # cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))
    logits = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(logits) * 100

    # % of correct answers found in batch
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    learning_rate = 0.003
    # optimizer = tf.train.GradientDescentOptimizer(0.003)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy)

    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        # load batch of images and correct answers
        batch_X, batch_Y = mnist.train.next_batch(100)
        train_data = {X: batch_X, Y_: batch_Y}

        # train
        sess.run(train_step, feed_dict=train_data)

    test_data = {X: mnist.test.images, Y_: mnist.test.labels}
    a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
    print(a)

if __name__ == "__main__":
    run()