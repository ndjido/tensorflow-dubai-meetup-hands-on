import os
import tensorflow as tf
import pandas as pd

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", "input", 'Input Directory')
flags.DEFINE_string("output_dir", "output", 'Output Directory')
flags.DEFINE_string("checkpoint_dir", "model", "Checkpoint Directory")
flags.DEFINE_integer('nb_iteration', 10000, "Number of Iterations")

TRAIN_SIZE = 89
TEST_SIZE = 60

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

def iris_parser(record):
    record_types = [tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32]
    line = tf.decode_csv(record, record_types, field_delim=',')
    return line


def read_csv(file_names, batch_size):
    examples_op = tf.contrib.learn.read_batch_examples(
        file_names,
        batch_size = batch_size,
        reader=tf.TextLineReader,
        parse_fn = iris_parser
    )

    # Important: convert examples to dict for ease of use in `input_fn`
    # Map each header to its respective column (COLUMNS order
    # matters!
    examples_dict_op = {}
    for i, header in enumerate(COLUMNS):
        examples_dict_op[header] = examples_op[:, i]

    return examples_dict_op

def run_trainer():
    class_names =['Setosa', 'Versicolour', 'Virginica']
    col_names = ['sepal_lenght', 'sepal_width', 'petal_length', 'petal_width', 'target']

    filename_queue_train = tf.train.string_input_producer([os.path.join(FLAGS.input_dir, 'iris_tain.csv')])
    filename_queue_test = tf.train.string_input_producer([os.path.join(FLAGS.input_dir, 'iris_test.csv')])

    train_reader = tf.TextLineReader()
    test_reader = tf.TextLineReader()
    _, value_train = train_reader.read(filename_queue_train)
    _, value_test = test_reader.read(filename_queue_test)

    record_types = [tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32]
    sl, sw, pl, pw, se, ve, vi = tf.decode_csv(value_train, record_types, field_delim=',')
    slT, swT, plT, pwT, seT, veT, viT = tf.decode_csv(value_test, record_types, field_delim=',')

    features_train = tf.stack([sl, sw, pl, pw])
    features_test = tf.stack([slT, swT, plT, pwT])
    label_train = tf.stack([se, ve, vi])
    label_test = tf.stack([seT, veT, viT])

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(TRAIN_SIZE):
            # Retrieve a single instance:
            example, label = sess.run([features, col5])

        coord.request_stop()
        coord.join(threads)


def main(_):
    try:
        run_trainer()
    except Exception as e:
        print(e.__traceback__)

if __name__ == '__main__':
    tf.app.run()
