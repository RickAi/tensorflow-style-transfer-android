import numpy as np
import tensorflow as tf
from ops import *
from tensorflow.python.tools import inspect_checkpoint as chkp

# TODO: 1. load the weights from trained model
# TODO: 2. a decoder to convert images back

ENCODER_LAYERS = {
    'vgg_16/conv1/conv1_1/biases': [64],
    'vgg_16/conv1/conv1_1/weights': [3, 3, 3, 64],
    'vgg_16/relu1_1': [],
    'vgg_16/conv1/conv1_2/biases': [64],
    'vgg_16/conv1/conv1_2/weights': [3, 3, 64, 64],
    'vgg_16/relu1_2': [],
    'vgg_16/pool1': [],

    'vgg_16/conv2/conv2_1/biases': [128],
    'vgg_16/conv2/conv2_1/weights': [3, 3, 64, 128],
    'vgg_16/relu2_1': [],
    'vgg_16/conv2/conv2_2/biases': [128],
    'vgg_16/conv2/conv2_2/weights': [3, 3, 128, 128],
    'vgg_16/relu2_2': [],
    'vgg_16/pool2': [],

    'vgg_16/conv3/conv3_1/biases': [256],
    'vgg_16/conv3/conv3_1/weights': [3, 3, 128, 256],
    'vgg_16/relu3_1': [],
    'vgg_16/conv3/conv3_2/biases': [256],
    'vgg_16/conv3/conv3_2/weights': [3, 3, 256, 256],
    'vgg_16/relu3_2': [],
    'vgg_16/conv3/conv3_3/biases': [256],
    'vgg_16/conv3/conv3_3/weights': [3, 3, 256, 256],
    'vgg_16/relu3_3': [],
    'vgg_16/pool3': [],

    'vgg_16/conv4/conv4_1/biases': [512],
    'vgg_16/conv4/conv4_1/weights': [3, 3, 256, 512],
    'vgg_16/relu4_1': [],
}

class Encoder(object):
    STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

    def __init__(self, weights_path):
        self.weight_vars = {}
        for name in ENCODER_LAYERS:
            if len(ENCODER_LAYERS[name]) > 0:
                self.weight_vars[name] = tf.get_variable(name, ENCODER_LAYERS[name], trainable=False)

        saver = tf.train.Saver(self.weight_vars)
        with tf.Session() as sess:
            saver.restore(sess, weights_path)

    def encode(self, image, is_train=False):
        exp = 6  # expansion ratio
        with tf.variable_scope('mobilenetv2'):
            net = conv2d_block(image, 32, 3, 2, is_train, name='conv1_1')  # size/2

            net = res_block(net, 1, 16, 1, is_train, name='res2_1')

            net = res_block(net, exp, 24, 2, is_train, name='res3_1')  # size/4
            net = res_block(net, exp, 24, 1, is_train, name='res3_2')

            net = res_block(net, exp, 32, 2, is_train, name='res4_1')  # size/8
            net = res_block(net, exp, 32, 1, is_train, name='res4_2')
            net = res_block(net, exp, 32, 1, is_train, name='res4_3')

            net = res_block(net, exp, 64, 1, is_train, name='res5_1')
            net = res_block(net, exp, 64, 1, is_train, name='res5_2')
            net = res_block(net, exp, 64, 1, is_train, name='res5_3')
            net = res_block(net, exp, 64, 1, is_train, name='res5_4')

            net = res_block(net, exp, 96, 2, is_train, name='res6_1')  # size/16
            net = res_block(net, exp, 96, 1, is_train, name='res6_2')
            net = res_block(net, exp, 96, 1, is_train, name='res6_3')

            net = res_block(net, exp, 160, 2, is_train, name='res7_1')  # size/32
            net = res_block(net, exp, 160, 1, is_train, name='res7_2')
            net = res_block(net, exp, 160, 1, is_train, name='res7_3')

            net = res_block(net, exp, 320, 1, is_train, name='res8_1', shortcut=False)

            net = pwise_block(net, 1280, is_train, name='conv9_1')
            net = global_avg(net)
            logits = flatten(conv_1x1(net, num_classes, name='logits'))

            pred = tf.nn.softmax(logits, name='prob')
            return logits, pred

    def preprocess(self, image, mode='BGR'):
        if mode == 'BGR':
            return image - np.array([103.939, 116.779, 123.68])
        else:
            return image - np.array([123.68, 116.779, 103.939])

    def deprocess(self, image, mode='BGR'):
        if mode == 'BGR':
            return image + np.array([103.939, 116.779, 123.68])
        else:
            return image + np.array([123.68, 116.779, 103.939])