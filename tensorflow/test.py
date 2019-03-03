from tensorflow.contrib import slim
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

image = tf.get_variable('image', [8, 256, 256, 3])
with tf.variable_scope('vgg_16', [image]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d]):
        net = slim.repeat(image, 2, slim.conv2d, 64, [3, 3], scope='conv1', trainable=False)
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2', trainable=False)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3', trainable=False)
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4', trainable=False)
init_fn = slim.assign_from_checkpoint_fn(
          'vgg_16.ckpt', slim.get_model_variables(), ignore_missing_vars=True)
print('hang')