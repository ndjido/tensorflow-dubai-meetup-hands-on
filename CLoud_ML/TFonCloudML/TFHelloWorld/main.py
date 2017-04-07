import tensorflow as tf


def run_training():
    x = tf.placeholder('float')
    w = tf.Variable(.5, 'weight')
    y = tf.multiply(x, w)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a = sess.run(y, feed_dict={x: 10})

        print(a)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
