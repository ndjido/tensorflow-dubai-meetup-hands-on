import tensorflow as tf


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


def create_conv_layer(inputs, inputs_channels_dim, patch_dim, output_channels_dim, stride_x=1, stride_y=1,
                      name="ConvLayer", activation=False):
    with tf.name_scope(name):
        W = tf.Variable(tf.random_normal([patch_dim, patch_dim, inputs_channels_dim, output_channels_dim], stddev=0.1),
                        name="W")
        variable_summaries(W, 'Weights')
        b = tf.Variable(tf.ones([output_channels_dim]), name="b")
        variable_summaries(b, 'Biases')

        layer = tf.nn.conv2d(inputs, W, strides=[1, stride_x, stride_y, 1], padding="SAME") + b
        if activation:
            return tf.nn.relu(layer)
        else:
            return layer


def create_polling_layer(conv_layer, _padding="SAME", stride_x=2, stride_y=2, name="PollingLayer"):
    with tf.name_scope(name):
        return tf.nn.relu(
            tf.nn.max_pool(conv_layer, ksize=[1, stride_x, stride_y, 1], strides=[1, stride_x, stride_y, 1],
                           padding=_padding))


def create_fully_connected_layer(inputs, inputs_dim, output_dim, name="fully_connected_layer", p_dropout=0):
    with tf.name_scope(name):
        W = tf.Variable(tf.random_normal([inputs_dim, output_dim], stddev=0.1), name="W")
        variable_summaries(W, 'Weights')
        b = tf.Variable(tf.ones([output_dim]) / tf.constant(10.0), name="b")
        variable_summaries(b, 'Biases')
        layer = tf.nn.softmax(tf.matmul(inputs, W) + b)
        if p_dropout > 0.0 and p_dropout < 1.0:
            return tf.nn.dropout(layer, p_dropout)
        else:
            return layer


def create_softmax_layer(inputs, inputs_dim, output_dim, name="softmax_layer"):
    with tf.name_scope(name):
        W = tf.Variable(tf.random_normal([inputs_dim, output_dim], stddev=0.1), name="W")
        variable_summaries(W, 'Weights')
        b = tf.Variable(tf.ones([output_dim]) / tf.constant(10.0), name="b")
        variable_summaries(b, 'Biases')
        return tf.nn.relu(tf.matmul(inputs, W) + b)
