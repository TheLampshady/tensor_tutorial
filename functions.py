# Initialize session
import tensorflow as tf


def init_tensor(sess):
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)

    # Place holders function like parameters of a function
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    # Adding another paramter
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    # Add print operation
    init = tf.global_variables_initializer()
    sess.run(init)

    # Needs a session to display
    print(W.eval())


def sample_tensor(sess):
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)  # also tf.float32 implicitly
    print(node1, node2)

    print(sess.run([node1, node2]))

    # Setting Variables
    var1 = tf.Variable(3.)
    var2 = tf.Variable(3, dtype=tf.float32, name="var2")

    print("Var1: ", var1)
    print("Var2: ", var2)

    place1 = tf.placeholder(tf.float32)
    place2 = tf.placeholder(tf.float32)
    adder_node = tf.add(place1, place2)

    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)

    init = tf.global_variables_initializer()
    sess.run(init)

    # Place holders function like parameters of a function
    # Tensor Operation
    linear_model = W * x + b

    # Placeholder for target values
    y = tf.placeholder(tf.float32)

    # Calculate loss by delta of output and targets.
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)

    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    # Manually adjust weights and bias to produce same target. No loss
    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])
    sess.run([fixW, fixb])
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess.run(init)  # reset values to incorrect defaults.
    for i in range(1000):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

    print(sess.run([W, b]))