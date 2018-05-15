import tensorflow as tf
import numpy as np
from data_preprocess import train_preprocess
from forward import forward

STEPS = 25000
BATCH_SIZE = 100


def backward():
    x = tf.placeholder(tf.float32, shape=[None, 12])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    train_x, train_y_ = train_preprocess()
    y, pred = forward(x)

    # 定义损失函数loss
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))

    # 定义反向传播方法
    train_step = tf.train.GradientDescentOptimizer(0.0003).minimize(loss)

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
            train_y_ = train_y_[index]
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


if __name__ == '__main__':
    backward()
