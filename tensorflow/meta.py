import numpy as np
from tensorflow.contrib import slim

from utils import list_images
from utils import get_train_images
import tensorflow as tf
import argparse
from datetime import datetime
from network_ops import upsample
from tensorflow.python.ops import init_ops
from collections import defaultdict
from keras import backend as K
from tensorflow.python.framework import ops

# meta network implementation
#
# pre:
# cd tool & bash download_vgg16.sh
# python meta.py
class VGG16(object):

    def encode(self, image, reuse=True):
        with tf.variable_scope('vgg_16', [image], reuse=reuse):
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME',
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                outs = []
                net = slim.repeat(image, 2, slim.conv2d, 64, [3, 3], scope='conv1', trainable=False)
                outs.append(net)
                net = slim.max_pool2d(net, [2, 2], scope='pool1')

                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2', trainable=False)
                outs.append(net)
                net = slim.max_pool2d(net, [2, 2], scope='pool2')

                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3', trainable=False)
                outs.append(net)
                net = slim.max_pool2d(net, [2, 2], scope='pool3')

                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4', trainable=False)
                outs.append(net)

                return outs

    def mean_std(self, features, epsilon=1e-5):
        mean_std_features = []
        for inputs in features:
            shape = inputs.shape
            inputs = (K.permute_dimensions(inputs, (0, 3, 1, 2)))
            inputs = K.reshape(inputs, (-1, int(shape[3]), int(shape[1]) * int(shape[2])))
            m = K.mean(inputs, axis=-1, keepdims=False)
            v = K.sqrt(K.var(inputs, axis=-1, keepdims=False) + K.constant(epsilon, dtype=inputs.dtype.base_dtype))
            mean_std_features.append(K.concatenate([m, v], axis=-1))
        mean_std_features = tf.concat(mean_std_features, -1)
        return mean_std_features


class MetaNet:

    def __init__(self, transform_param):
        self.transform_param = transform_param

    def encode(self, features):
        with tf.variable_scope('Meta', [features]):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d]):
                net = slim.linear(features, 128 * len(self.transform_param), scope='hidden')
                net = slim.relu(net, 128 * len(self.transform_param), scope='relu')
                filters = {}
                for i, (name, params) in enumerate(self.transform_param.items()):
                    filters[name] = slim.linear(net[:, i * 128:(i + 1) * 128], params,
                                                scope='fc{}'.format(i + 1))
        return filters


class TransformNet:

    def __init__(self, base=32):
        self.base = base
        self.weights = {}

    def encode(self, weights, reuse=True):
        net = weights
        with tf.variable_scope('TransformNet', [net], reuse=reuse):
            with tf.variable_scope('downsampling', [net]):
                net = self.conv_layer(net, self.base, 'conv_layer1', kernel_size=9, trainable=True)
                net = self.conv_layer(net, self.base * 2, 'conv_layer2', kernel_size=3, stride=2)
                net = self.conv_layer(net, self.base * 4, 'conv_layer3', kernel_size=3, stride=2)
            with tf.variable_scope('residuals', [net]):
                net = self.residual_block(net, self.base * 4, 'residual_block1')
                net = self.residual_block(net, self.base * 4, 'residual_block2')
                net = self.residual_block(net, self.base * 4, 'residual_block3')
                net = self.residual_block(net, self.base * 4, 'residual_block4')
                net = self.residual_block(net, self.base * 4, 'residual_block5')
            with tf.variable_scope('upsampling', [net]):
                net = self.conv_layer(net, self.base * 2, 'conv_layer1', kernel_size=3, upsample_factor=2)
                net = self.conv_layer(net, self.base, 'conv_layer2', kernel_size=3, upsample_factor=2)
                net = self.conv_layer(net, 3, 'conv_layer3', kernel_size=9, instance_norm=False, relu=False,
                                      trainable=True)
        return net

    def get_param_dict(self):
        variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='TransformNet')
        param_dict = defaultdict(int)
        for variable in variables:
            name = variable.name
            if 'myconv' in name and 'weights' in name:
                param_dict[name.rsplit('/', 1)[0]] += int(np.prod(variable.shape))

            if 'myconv' in name and 'biases' in name:
                param_dict[name.rsplit('/', 1)[0]] += int(np.prod(variable.shape))

        return param_dict

    def get_myconv_variables(self):
        variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='TransformNet')
        variable_dict = defaultdict(int)
        for variable in variables:
            name = variable.name
            if 'myconv' in name:
                variable_dict[name] = variable

        return variable_dict

    def set_weights(self, weights):
        variables = self.get_myconv_variables()
        for name, params in weights.items():
            weights_name = name + '/weights:0'
            biases_name = name + '/biases:0'

            weight_len = np.prod(variables[weights_name].shape)
            self.weights[weights_name] = tf.assign(variables[weights_name], tf.reshape(weights[name][0][:weight_len], variables[weights_name].shape))
            self.weights[biases_name] = tf.assign(variables[biases_name], tf.reshape(weights[name][0][weight_len:], variables[biases_name].shape))

    def get_weights(self, name):
        weights_name = 'TransformNet/' + name.split("/", 1)[1] + '/myconv/weights:0'
        biases_name = 'TransformNet/' + name.split("/", 1)[1] + '/myconv/biases:0'
        variables = {}
        if weights_name in self.weights and biases_name in self.weights:
            variables['weights'] = self.weights[weights_name]
            variables['biases'] = self.weights[biases_name]
        return variables

    def conv_layer(self, inputs, out_channels, scope, kernel_size=3, stride=1, upsample_factor=None, instance_norm=True,
                   relu=True, trainable=False):
        with tf.variable_scope(scope, [inputs, out_channels, kernel_size, stride,
                                       upsample_factor, instance_norm, relu, trainable]):
            net = inputs
            if upsample_factor:
                net = upsample(net, upsample_factor, scope='upsample')

            padding = kernel_size // 2
            net = tf.pad(net, paddings=[[0, 0], [padding, padding], [padding, padding], [0, 0]], mode="REFLECT")
            if trainable:
                net = slim.conv2d(net, out_channels, [kernel_size, kernel_size], stride, scope='conv',
                                  activation_fn=None, padding='VALID')
            else:
                variables = self.get_weights(tf.contrib.framework.get_name_scope())
                if len(variables) > 0:
                    net = self.my_conv(net, variables['weights'], variables['biases'], stride, padding='VALID', name='myconv')
                else:
                    net = slim.conv2d(net, out_channels, [kernel_size, kernel_size], stride, trainable=trainable,
                                      scope='myconv', activation_fn=None, padding='VALID',
                                      weights_initializer=init_ops.zeros_initializer(),
                                      biases_initializer=init_ops.zeros_initializer())

            if instance_norm:
                net = tf.contrib.layers.instance_norm(net, scope='instance_norm')
            if relu:
                net = slim.relu(net, out_channels, scope='relu')

            return net

    def my_conv(self, input, filter, bias, strides, padding, name):
        with ops.name_scope(name, [input, filter, bias, strides, padding]):
            net = tf.nn.conv2d(input, filter, [1, strides, strides, 1], padding, name='weights')
            net = tf.nn.bias_add(net, bias, name='biases')
        return net

    def residual_block(self, inputs, out_channels, scope):
        with tf.variable_scope(scope, [inputs, out_channels]):
            net = inputs
            net = self.conv_layer(net, out_channels, 'conv_layer1', kernel_size=3, stride=1)
            net = self.conv_layer(net, out_channels, 'conv_layer2', kernel_size=3, stride=1, relu=False)
            return net + inputs


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


EPOCHS = 1000
LEARNING_RATE = 1e-3
STYLE_WEIGHT = 50
TV_WEIGHT = 1e-6
BATCH_SIZE = 8
TRAINING_IMAGE_SHAPE = (256, 256, 3)  # (height, width, color_channels)

MODEL_SAVE_PATH = "./meta_models/meta_models.ckpt"
LOG_SAVE_PATH = "./meta_logs"
LOGGING_PERIOD = 20

DEMO_CONTENT_DIR = './images/board_content'
DEMO_STYLE_DIR = './images/board_style'

VGG16_WEIGHT_PATH = 'vgg_16.ckpt'

if __name__ == '__main__':
    start_time = datetime.now()
    print('\n>>> Begin to stylize images with style weight: %.2f\n' % STYLE_WEIGHT)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--provision", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="y for server n for local")
    parser.add_argument("--training", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="y for training n for infer")
    args = parser.parse_args()

    TRAINING_CONTENT_DIR = '/research/dept3/ybai/datasets/COCO/train' if args.provision else './images/content'
    TRAINING_STYLE_DIR = '/research/dept3/ybai/datasets/WikiArt/train_1' if args.provision else './images/style'

    print('\n>> The content database is %s' % TRAINING_CONTENT_DIR)
    print('\n>> The style database is %s' % TRAINING_STYLE_DIR)

    content_imgs_path = list_images(TRAINING_CONTENT_DIR)
    style_imgs_path = list_images(TRAINING_STYLE_DIR)

    num_imgs = min(len(content_imgs_path), len(style_imgs_path))
    content_imgs_path = content_imgs_path[:num_imgs]
    style_imgs_path = style_imgs_path[:num_imgs]

    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

    demo_content_images = get_train_images(list_images(DEMO_CONTENT_DIR), crop_height=HEIGHT, crop_width=WIDTH)
    demo_style_images = get_train_images(list_images(DEMO_STYLE_DIR), crop_height=HEIGHT, crop_width=WIDTH)

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        content = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='content')
        style = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')

        ## -------init------
        vgg = VGG16()
        vgg.encode(style, reuse=False)
        # load weights from pre-trained model
        restore_fn = slim.assign_from_checkpoint_fn(
            VGG16_WEIGHT_PATH, slim.get_model_variables('vgg_16'), ignore_missing_vars=True)

        transform_net = TransformNet()
        transform_net.encode(content, reuse=False)
        # ---------init operation, encodes is no sense

        style_features = vgg.encode(style)
        style_features_mean = vgg.mean_std(style_features)

        # generate weights
        metanet = MetaNet(transform_net.get_param_dict())
        style_weights = metanet.encode(style_features_mean)

        # set weights and transform
        transform_net = TransformNet()
        transform_net.set_weights(style_weights)
        transformed_images = transform_net.encode(content)

        # comparision
        content_features = vgg.encode(content)
        transformed_features = vgg.encode(transformed_images)

        content_loss = tf.losses.mean_squared_error(transformed_features[2], content_features[2])
        style_loss = STYLE_WEIGHT * tf.losses.mean_squared_error(vgg.mean_std(transformed_features), style_features_mean)
        y = transformed_images
        tv_loss = TV_WEIGHT * (tf.reduce_sum(tf.math.abs(y[:, :, :-1, :] - y[:, :, 1:, :])) +
                               tf.reduce_sum(tf.math.abs(y[:, :-1, :, :] - y[:, 1:, :, :])))
        total_loss = content_loss + style_loss + tv_loss

        cl_scalar = tf.summary.scalar("content_loss", content_loss)
        sl_scalar = tf.summary.scalar("style_loss", STYLE_WEIGHT * style_loss)
        tl_scalar = tf.summary.scalar("total_loss", total_loss)
        loss_summary_op = tf.summary.merge([cl_scalar, sl_scalar, tl_scalar])

        ci_image = tf.summary.image("content_images", content, max_outputs=8)
        si_image = tf.summary.image("style_images", style, max_outputs=8)
        demo_image_op = tf.summary.merge([ci_image, si_image])

        ti_image = tf.summary.image("transfer_images", transformed_images, max_outputs=8)

        global_step = tf.Variable(0, trainable=False)
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(total_loss, global_step=global_step)

        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(LOG_SAVE_PATH, graph=tf.get_default_graph())

        step = 0
        n_batches = int(len(content_imgs_path) // BATCH_SIZE)
        saver = tf.train.Saver(max_to_keep=10)

        for epoch in range(EPOCHS):
            np.random.shuffle(content_imgs_path)
            np.random.shuffle(style_imgs_path)

            for batch in range(n_batches):
                content_batch_path = content_imgs_path[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]
                style_batch_path = style_imgs_path[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]

                content_batch = get_train_images(content_batch_path, crop_height=HEIGHT, crop_width=WIDTH)
                style_batch = get_train_images(style_batch_path, crop_height=HEIGHT, crop_width=WIDTH)

                sess.run(train_op, feed_dict={content: content_batch, style: style_batch})

                step += 1

                if step % 1000 == 0:
                    saver.save(sess, MODEL_SAVE_PATH, global_step=step)

                is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                if is_last_step or step == 1 or step % LOGGING_PERIOD == 0:
                    elapsed_time = datetime.now() - start_time
                    _content_loss, _style_loss, _loss, loss_summary = \
                        sess.run([content_loss, style_loss, total_loss, loss_summary_op],
                                 feed_dict={content: content_batch, style: style_batch})

                    summary_writer.add_summary(loss_summary, step * BATCH_SIZE + batch)
                    print('step: %d,  total loss: %.3f,  elapsed time: %s' % (step, _loss, elapsed_time))
                    print('content loss: %.3f' % (_content_loss))
                    print('style loss  : %.3f,  weighted style loss: %.3f\n' % (
                        _style_loss, STYLE_WEIGHT * _style_loss))

                    # add transfer images into board
                    transfer_summary = sess.run(ti_image,
                                                feed_dict={content: demo_content_images,
                                                           style: demo_style_images})
                    summary_writer.add_summary(transfer_summary, step * BATCH_SIZE + batch)

                if step == 1:
                    demo_summary = sess.run(demo_image_op,
                                            feed_dict={content: demo_content_images, style: demo_style_images})
                    summary_writer.add_summary(demo_summary, step * BATCH_SIZE + batch)

        ###### Done Training & Save the model ######
        saver.save(sess, MODEL_SAVE_PATH)

        elapsed_time = datetime.now() - start_time
        print('Done training! Elapsed time: %s' % elapsed_time)
        print('Model is saved to: %s' % MODEL_SAVE_PATH)
        print("\nRun the command line:\n" \
              "--> tensorboard --logdir=./logs " \
              "\nThen open http://0.0.0.0:6006/ into your web browser")
