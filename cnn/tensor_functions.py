import tensorflow as tf
import numpy as np
import scipy.misc as misc


def variable_summaries(var, histogram_name='histogram'):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :type var: tf.Variable
    :type histogram_name: str
    :rtype: tf.Tensor
    """
    mean = tf.reduce_mean(var)
    mean_scalar = tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    stddev_scalar = tf.summary.scalar('stddev', stddev)
    max_scalar = tf.summary.scalar('max', tf.reduce_max(var))
    min_scalar = tf.summary.scalar('min', tf.reduce_min(var))

    histogram = tf.summary.histogram(histogram_name, var)

    return tf.summary.merge([
        mean_scalar,
        stddev_scalar,
        max_scalar,
        min_scalar,
        histogram
    ])


def weight_variable(shape, stddev=0.1, enable_summary=True):
    """
    Create a weight variable with appropriate initialization.
    :type shape: list <float>
    :type stddev: float
    :type enable_summary: bool
    :rtype: tf.Variable
    """
    name = 'Weights'
    initial = tf.truncated_normal(shape, stddev=stddev, name="%s_Init" % name)
    weight = tf.Variable(initial, name=name)
    with tf.name_scope(name):
        if enable_summary:
            variable_summaries(weight, name)

    return weight


def bias_variable(shape, init=0.1, enable_summary=True):
    """
    Create a bias variable with appropriate initialization.
    :type shape: list <float>
    :type init: float
    :type enable_summary: bool
    :rtype: tf.Variable
    """
    name = 'Biases'
    initial = tf.constant(init, shape=shape, name="%s_Init" % name)
    biases = tf.Variable(initial, name=name)
    with tf.name_scope(name):
        if enable_summary:
            variable_summaries(biases, name)

    return biases


def build_sprite(sprite_images, filename='sprite_1024.png'):
    x = None
    res = None
    for i in range(32):
        x = None
        for j in range(32):
            img = sprite_images[i * 32 + j, :].reshape((28, 28))
            x = np.concatenate((x, img), axis=1) if x is not None else img
        res = np.concatenate((res, x), axis=0) if res is not None else x

    misc.toimage(256 - res, channel_axis=0).save(filename)


def build_labels(labels, filename='labels_1024.tsv'):
    label_file = open(filename, 'w')
    for target in labels:
        value = int(np.where(target == 1)[0])
        label_file.write("%d\n" % value)
    label_file.close()
