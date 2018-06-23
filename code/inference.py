import tensorflow as tf


# 神经网络相关参数
INPUT_NODE = 22
OUTPUT_NODE = 1
LAYER1_NODE = 14


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights",
        shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1))

    # 正则化
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    # 声明第一层神经网络，并完成前向传播
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable(
            "biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 声明第二层神经网络，并完成前向传播
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
    layer2 = tf.matmul(layer1, weights) + biases

    return layer2
