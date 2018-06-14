import tensorflow as tf

REGULARIZER = 0.1


def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses',
                         tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.random_normal(shape))
    return b


def forward(x):
    w1 = get_weight([22, 44], REGULARIZER)
    y1 = tf.sigmoid(tf.matmul(x, w1))

    w2 = get_weight([44, 1], REGULARIZER)
    b = get_bias([1])
    y = tf.matmul(y1, w2) + b
    pred = tf.cast(tf.sigmoid(y) > 0.5, tf.float32)
    return y, pred
