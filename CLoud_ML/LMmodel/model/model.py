import os
import tensorflow as tf
from tensorflow.python.util import compat
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("output_dir", "output", 'Output Directory')


def train():
    sample_size = 1000
    a = np.pi
    b = 7
    x_data = np.random.randn(sample_size)
    noise = [n * np.random.normal(0.0, 0.1) for n in np.ones(sample_size)]
    y_data = (a * x_data) + (np.ones(sample_size) * b) + noise

    with tf.name_scope('Model'):
        W = tf.Variable(np.random.normal(0.0, 1), name="W")
        b = tf.Variable(tf.zeros([1]), name='b')
        y_model = tf.multiply(W, x_data) + b

    with tf.name_scope('training'):
        loss_func = tf.reduce_mean(tf.square(y_model - y_data))
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        trainer = optimizer.minimize(loss_func)

    _input = tf.placeholder(tf.float32, shape=None)
    output = W * _input + b

    nb_iterations = 10

    merged_summary = tf.summary.merge_all()

    saved_model_path = os.path.join(FLAGS.output_dir, 'model', 'lm_model_v3')
    model_builder = tf.python.saved_model.builder.SavedModelBuilder(saved_model_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(os.path.join(FLAGS.output_dir, 'summary'), sess.graph)

        for i in range(nb_iterations):
            sess.run(trainer)
            y_estimated = sess.run(W) * x_data + sess.run(b)
            print(y_estimated)
            #summary, _ = sess.run([merged_summary])
            #writer.add_summary(summary)

            # Saving the model
        model_builder.add_meta_graph_and_variables(
            sess,
            [tf.python.saved_model.tag_constants.SERVING],
            signature_def_map={
                "magic_model": tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs={"input": _input},
                    outputs={"output": output})
            })
        model_builder.save()

def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
