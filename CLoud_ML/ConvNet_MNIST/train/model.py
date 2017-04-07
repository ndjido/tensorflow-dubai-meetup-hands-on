import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from . import convnet_helpers as H

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", "input", 'Input Directory')
flags.DEFINE_string("output_dir", "output", 'Output Directory')
flags.DEFINE_string("checkpoint_dir", "model", "Checkpoint Directory")
flags.DEFINE_integer('epoch_size', 10000, "Number of EPOCHs")
flags.DEFINE_integer('batch_size', 100, "Batch Size")

MNIST_DIR = os.path.join(FLAGS.input_dir, 'MNIST_data')
SUMMARY_TRAIN = os.path.join(FLAGS.output_dir, 'summary_train')
SUMMARY_TEST = os.path.join(FLAGS.output_dir, 'summary_test')


def train():
    # download MNIST data
    mnist = input_data.read_data_sets(MNIST_DIR, one_hot=True)

    # Input Layers Dimensions
    A = 6
    B = 12
    C = 24
    D = 200

    nb_pixels_x = 28
    nb_pixels_y = nb_pixels_x
    _inputs_channels_dim = 1
    _output_dim = 10

    X = tf.placeholder(tf.float32, shape=[None, nb_pixels_x * nb_pixels_y])
    X_image = tf.reshape(X, [-1, nb_pixels_x, nb_pixels_y, _inputs_channels_dim])
    Y_ = tf.placeholder(tf.float32, shape=[None, _output_dim])

    Y1 = H.create_conv_layer(X_image, _inputs_channels_dim, 5, A, 1, 1, name="ConvLayer_A", activation=True)
    Y2 = H.create_conv_layer(Y1, A, 5, B, 2, 2, name="ConvLayer_B", activation=True)
    Y3 = H.create_conv_layer(Y2, B, 4, C, 2, 2, name="ConvLayer_C", activation=True)

    Y3_ = tf.reshape(Y3, shape=[-1, 7 * 7 * C])
    Y4 = H.create_fully_connected_layer(Y3_, 7 * 7 * C, D, name="fully_connected_layer", p_dropout=.7)

    Y = H.create_softmax_layer(Y4, D, _output_dim, name="softmax_layer")

    with tf.name_scope('cross_entropy'):
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=Y_))

    tf.summary.scalar("cross_entropy_loss", cross_entropy_loss)

    # Optimizer
    with tf.name_scope("Trainer"):
        trainer = tf.train.AdamOptimizer(.1).minimize(cross_entropy_loss)

    # Accuracy
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('accuracy', accuracy)

    # model saver
    saver = tf.train.Saver()
    checkpoint_file = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')

    # run the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # Summary writer for TensorBoad
        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(SUMMARY_TRAIN, sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARY_TEST)

        for i in range(10000):
            # train
            x_data, y_data = mnist.train.next_batch(100)
            summary, _ = sess.run([merged_summary, trainer], feed_dict={X: x_data, Y_: y_data})
            train_writer.add_summary(summary, i)
            # test
            summary, acc = sess.run([merged_summary, accuracy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
            test_writer.add_summary(summary, i)
            print('Step %s -> Accuracy %s' % (i, acc))

            # save the model
            saver.save(sess, checkpoint_file)


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
