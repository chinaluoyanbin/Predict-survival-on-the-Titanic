import tensorflow as tf


def get_weight(shape):
    w = tf.Variable(tf.random_normal(shape))
    return w


def get_bias(shape):
    b = tf.Variable(tf.random_normal(shape))
    return b


def forward(x):
    w1 = get_weight([12, 3])
    w2 = get_weight([3, 1])
    b = get_bias([1])

    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2) + b
    pred = tf.cast(tf.sigmoid(y) > 0.5, tf.float32)

    return y, pred
