import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name , stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram('histogram/' + name, var)


def run_trainer():
    # download MNIST data
    mnist = input_data.read_data_sets(MNIST_DIR, one_hot=True)

    # inputs
    x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # model
    with tf.name_scope("Model"):
        W = tf.Variable(tf.zeros([28 * 28, 10]), name="W")
        variable_summaries(W, 'Weights')
        b = tf.Variable(tf.zeros([10]), name="b")
        variable_summaries(b, 'Biases')
        y_model = tf.matmul(x, W) + b

    # loss function
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_model, labels=y_))
    tf.summary.scalar("cross_entropy", cross_entropy)

    # trainer
    with tf.name_scope("Trainer"):
        trainer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # Performance: Accuracy
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # model saver
    saver = tf.train.Saver()
    checkpoint_file = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')

    # Start training
    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Summary writer for TensorBoard
        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(SUMMARY_TRAIN, sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARY_TEST)


        # Training
        for i in range(FLAGS.epoch_size):
            x_data, y_data = mnist.train.next_batch(FLAGS.batch_size)
            summary, _ = sess.run([merged_summary, trainer], feed_dict={x: x_data, y_: y_data})
            train_writer.add_summary(summary, i)
            summary, acc = sess.run([merged_summary, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            test_writer.add_summary(summary, i)
            print('Step %s -> Accuracy %s' % (i, acc))

            # save the model
            saver.save(sess, checkpoint_file)


def main(_):
    run_trainer()


if __name__ == '__main__':
    tf.app.run()
