import tensorflow as tf
import input_data
import inference
import train
import pandas as pd


def prediction(dataset):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(
            tf.float32, [None, inference.INPUT_NODE], name='x-input')

        y = inference.inference(x, None)
        prediction = tf.cast(tf.sigmoid(y) > 0.5, tf.int32)
        # probability = tf.sigmoid(y)

        prediction_feed = {x: dataset['test_x']}
        variable_averages = tf.train.ExponentialMovingAverage(
            train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                submit = pd.concat(
                    [
                        dataset['test_y'],
                        pd.DataFrame(
                            sess.run(prediction, feed_dict=prediction_feed),
                            columns=['Survived'])
                    ],
                    axis=1)
                submit.to_csv('../output/submit.csv', index=False)
                print('----------------------------------')
                print('Predict and output successfully...')
                print('----------------------------------')
            else:
                print('---------------------------')
                print('No checkpoint file found...')
                print('---------------------------')


def main(argv=None):
    dataset = input_data.read_data_sets()
    prediction(dataset)


if __name__ == '__main__':
    tf.app.run()
