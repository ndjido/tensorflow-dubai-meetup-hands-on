import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('number', 123456789, "The number which sqrt to be computed!")
flags.DEFINE_string("checkpoint_dir", "output/model", "Checkpoint Directory")


def train():
    epsilon = tf.constant(1e-6)

    X = tf.cast(FLAGS.number, 'float32')
    Un_1 = tf.Variable(X, name="Number")
    Un = tf.assign(Un_1, .5 * tf.add(Un_1, (X / Un_1)))

    with tf.Session() as sess:
        # Initialization
        sess.run(tf.global_variables_initializer())

        # run model
        sqrt_x = None
        while (sess.run(Un_1) - sess.run(Un)) > sess.run(epsilon):
            sqrt_x = sess.run(Un)
            print(sqrt_x)


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
