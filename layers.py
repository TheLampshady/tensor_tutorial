import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


def run():
    # Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
    mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

    X = tf.placeholder(tf.float32, [None, 28, 28, 1])

    W1 = tf.Variable(tf.truncated_normal([28*28, 200], stddev=0.1))
    W2 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
    B1 = tf.Variable(tf.zeros([200]))
    B2 = tf.Variable(tf.zeros([10]))

    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    # model
    XX = tf.reshape(X, [-1, 28 * 28])

    # Activation Function
    Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
    Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)

    # Regressino function
    # Y = tf.nn.softmax(tf.matmul(XX), W) + b)
    Y = tf.nn.softmax(tf.matmul(Y1, W2) + B2)


    # placeholder for correct labels
    Y_ = tf.placeholder(tf.float32, [None, 10])

    # loss function
    cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

    # % of correct answers found in batch
    is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(0.003)
    train_step = optimizer.minimize(cross_entropy)

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