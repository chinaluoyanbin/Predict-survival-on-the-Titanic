import tensorflow as tf
import numpy as np
import pandas as pd
from data_preprocess import train_preprocess
from data_preprocess import test_preproces
from forward import forward

STEPS = 25000
BATCH_SIZE = 100
LEARNING_RATE = 0.001
DECAY_STEPS = 222750
DECAY_RATE = 0.96


def backward():
    x = tf.placeholder(tf.float32, shape=[None, 22])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    train_x, train_y_ = train_preprocess()
    y, pred = forward(x)

    # 定义交叉熵损失函数cross_entropy_loss
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
    tf.add_to_collection('losses', cross_entropy_loss)

    # 定义总的损失函数loss
    loss = tf.add_n(tf.get_collection('losses'))

    # 定义指数衰减学习率
    global_step = tf.Variable(0, trainable=False)
    decayed_learning_rate = tf.train.exponential_decay(
        LEARNING_RATE, global_step, DECAY_STEPS, DECAY_RATE)

    # 定义反向传播方法
    train_step = tf.train.GradientDescentOptimizer(
        decayed_learning_rate).minimize(
            loss, global_step=global_step)

    # 定义准确率
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, y_), tf.float32))

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        train_loss = []
        train_acc = []

        for i in range(STEPS):
            index = np.random.permutation(len(train_y_))
            train_x = train_x.take(index)
            train_y_ = train_y_.take(index)
            for j in range(len(train_y_) // 100 + 1):
                start = j * BATCH_SIZE
                end = start + BATCH_SIZE
                sess.run(
                    train_step,
                    feed_dict={
                        x: train_x[start:end],
                        y_: train_y_[start:end]
                    })
            if i % 1000 == 0:
                train_loss_temp = sess.run(
                    loss,
                    feed_dict={
                        x: train_x[start:end],
                        y_: train_y_[start:end]
                    })
                train_loss.append(train_loss_temp)
                train_acc_temp = sess.run(
                    accuracy,
                    feed_dict={
                        x: train_x[start:end],
                        y_: train_y_[start:end]
                    })
                train_acc.append(train_acc_temp)
                print(train_loss_temp, ' ', train_acc_temp)
                print('global_step:', sess.run(global_step),
                      ' ', 'decayed_learning_rate:',
                      sess.run(decayed_learning_rate))

        # 使用训练好的模型，喂入测试集数据
        test_x = test_preproces()
        test = pd.read_csv('../input/test.csv')
        result = sess.run(pred, feed_dict={x: test_x})
        # print('----------------')
        # print('Prediction:')
        # print(result)
        submit = test[['PassengerId']]
        submit.insert(1, 'Survived', result)
        submit['Survived'] = submit['Survived'].astype(np.int32)
        submit.to_csv('../output/submit.csv')


if __name__ == '__main__':
    backward()
