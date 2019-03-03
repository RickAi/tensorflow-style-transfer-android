# Encoder is fixed to the first few layers (up to relu4_1)
# of VGG-11 (pre-trained on ImageNet)
# This code is a modified version of Anish Athalye's vgg.py
# https://github.com/anishathalye/neural-style/blob/master/vgg.py

import numpy as np
import tensorflow as tf

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
    STYLE_LAYERS = ('vgg_16/relu1_1', 'vgg_16/relu2_1', 'vgg_16/relu3_1', 'vgg_16/relu4_1')

    def __init__(self, weights_path):
        self.weight_vars = {}
        for name in ENCODER_LAYERS:
            if len(ENCODER_LAYERS[name]) > 0:
                self.weight_vars[name] = tf.get_variable(name, ENCODER_LAYERS[name], trainable=False)

        saver = tf.train.Saver(self.weight_vars)
        with tf.Session() as sess:
            saver.restore(sess, weights_path)


    def encode(self, image):
        # create the computational graph
        layers = {}
        current = image

        kernel, bias = None, None
        for layer in ENCODER_LAYERS:
            kind = layer.split("/")[1][:4]

            if kind == 'conv':
                if layer.split("/")[-1] == 'biases':
                    bias = self.weight_vars[layer]
                    continue
                kernel = self.weight_vars[layer]
                current = conv2d(current, kernel, bias)

            elif kind == 'relu':
                current = tf.nn.relu(current)

            elif kind == 'pool':
                current = pool2d(current)

            layers[layer] = current

        # assert (len(layers) == len(ENCODER_LAYERS))

        enc = layers[list(ENCODER_LAYERS.keys())[-1]]

        return enc, layers

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


def conv2d(x, kernel, bias):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

    return out


def pool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
