import tensorflow as tf
import inference
import numpy as np
# from pandas import DataFrame
import os
import input_data
import warnings

warnings.filterwarnings('ignore')

# 神经网络参数
STEPS = 25000
BATCH_SIZE = 100
REGULARIZER_RATE = 0.001
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.96

# 模型保存的路径和名称
MODEL_SAVE_PATH = '../model'
MODEL_NAME = 'model.ckpt'


# 神经网络训练模型
def train(dataset):
    # 定义输入输出placeholder
    x = tf.placeholder(
        tf.float32, [None, inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(
        tf.float32, [None, inference.OUTPUT_NODE], name='y-input')

    # 神经网络优化
    # 正则化类
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)

    y = inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 滑动平均模型
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # softmax交叉熵
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 指数衰减学习率
    DECAY_STEPS = int(STEPS * len(dataset['train_x']) / BATCH_SIZE)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, DECAY_STEPS, LEARNING_RATE_DECAY)

    # 梯度下降法
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 准确率
    prediction = tf.cast(tf.sigmoid(y) > 0.5, tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_), tf.float32))

    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(STEPS):
            index = np.random.permutation(len(dataset['train_y_']))
            xs = dataset['train_x'].take(index)
            ys = dataset['train_y_'][index]
            for j in range(len(dataset['train_y_']) // 100 + 1):
                start = j * BATCH_SIZE
                end = start + BATCH_SIZE
                sess.run(
                    train_op, feed_dict={
                        x: xs[start:end],
                        y_: ys[start:end]
                    })
            if i % 1000 == 0:
                step, loss_value, accuracy_value = sess.run(
                    [global_step, loss, accuracy],
                    feed_dict={
                        x: xs[start:end],
                        y_: ys[start:end]
                    })
                print(
                    "After %d training steps, loss on training batch is %f, accuracy is %f"
                    % (step, loss_value, accuracy_value))
                saver.save(
                    sess,
                    os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                    global_step=global_step)


def main(argv=None):
    dataset = input_data.read_data_sets()
    train(dataset)


if __name__ == '__main__':
    tf.app.run()
