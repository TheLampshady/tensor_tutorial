import tensorflow as tf


def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :type var: tf.Variable
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def weight_variable(shape, stddev=0.1, enable_summary=True):
    """
    Create a weight variable with appropriate initialization.
    :type shape: list <float>
    :type stddev: float
    :type enable_summary: bool
    :rtype: tf.Variable
    """
    with tf.name_scope('Weights'):
        initial = tf.truncated_normal(shape, stddev=stddev, name="Weight_Init")
        weight = tf.Variable(initial, name="Weight")
        if enable_summary:
            variable_summaries(weight)
        return weight


def bias_variable(shape, init=0.1, enable_summary=True):
    """
    Create a bias variable with appropriate initialization.
    :type shape: list <float>
    :type init: float
    :type enable_summary: bool
    :rtype: tf.Variable
    """
    with tf.name_scope('Biases'):
        initial = tf.constant(init, shape=shape, name="Bias_Init")
        bias = tf.Variable(initial, name="Bias")
        if enable_summary:
            variable_summaries(bias)
        return bias
