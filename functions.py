import tensorflow as tf


def variable_summaries(var, histogram_name='histogram'):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :type var: tf.Variable
    :type histogram_name: str
    """
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))

    tf.summary.histogram(histogram_name, var)


def weight_variable(shape, stddev=0.1, enable_summary=True):
    """
    Create a weight variable with appropriate initialization.
    :type shape: list <float>
    :type stddev: float
    :type enable_summary: bool
    :rtype: tf.Variable
    """
    name = 'Weights'
    with tf.name_scope('Weights'):
        initial = tf.truncated_normal(shape, stddev=stddev, name="%s_Init" % name)
        weight = tf.Variable(initial, name=name)
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
    with tf.name_scope(name):
        initial = tf.constant(init, shape=shape, name="%s_Init" % name)
        biases = tf.Variable(initial, name=name)
        if enable_summary:
            variable_summaries(biases, name)
        return biases
