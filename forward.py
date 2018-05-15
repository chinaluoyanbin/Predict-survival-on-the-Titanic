import tensorflow as tf


def get_weight(shape):
    w = tf.Variable(tf.random_normal(shape))
    return w


def get_bias(shape):
    b = tf.Variable(tf.random_normal(shape))
    return b


def forward(x):
    w = get_weight([12, 1])
    b = get_bias([1])
    y = tf.matmul(x, w) + b
    pred = tf.cast(tf.sigmoid(y) > 0.5, tf.float32)

    return y, pred
