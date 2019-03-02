import numpy as np
import tensorflow as tf
from ops import *

# TODO: 1. load the weights from trained model
# TODO: 2. a decoder to convert images back

class Encoder(object):
    STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

    def __init__(self, weights_path):
        pass

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