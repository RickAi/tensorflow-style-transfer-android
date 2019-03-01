# Demo - train the style transfer network & use it to generate an image

from __future__ import print_function

from train import train
from infer import stylize
from utils import list_images

import argparse

PROVISION = False
IS_TRAINING = False

# for training
ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'
LOGGING_PERIOD = 20

STYLE_WEIGHTS = [2.0]
MODEL_SAVE_PATHS = [
    'models/style_weight_2e0.ckpt',
]

# for inferring (stylize)
INFERRING_CONTENT_DIR = 'images/content'
INFERRING_STYLE_DIR = 'images/style'
OUTPUTS_DIR = 'outputs'


def main(args):
    TRAINING_CONTENT_DIR = '/research/dept3/ybai/datasets/COCO/train' if args.provision else './images/content'
    TRAINING_STYLE_DIR = '/research/dept3/ybai/datasets/WikiArt/train_1' if args.provision else './images/style'

    if args.training:

        content_imgs_path = list_images(TRAINING_CONTENT_DIR)
        style_imgs_path   = list_images(TRAINING_STYLE_DIR)
        print('\n>> The content database is %s' % TRAINING_CONTENT_DIR)
        print('\n>> The style database is %s' % TRAINING_STYLE_DIR)

        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\n>>> Begin to train the network with the style weight: %.2f\n' % style_weight)

            train(style_weight, content_imgs_path, style_imgs_path, ENCODER_WEIGHTS_PATH, 
                  model_save_path, logging_period=LOGGING_PERIOD, debug=True)

        print('\n>>> Successfully! Done all training...\n')

    else:

        content_imgs_path = list_images(INFERRING_CONTENT_DIR)
        style_imgs_path   = list_images(INFERRING_STYLE_DIR)

        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\n>>> Begin to stylize images with style weight: %.2f\n' % style_weight)

            stylize(content_imgs_path, style_imgs_path, OUTPUTS_DIR, 
                    ENCODER_WEIGHTS_PATH, model_save_path, 
                    suffix='-' + str(style_weight))

        print('\n>>> Successfully! Done all stylizing...\n')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--provision", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="y for server n for local")
    parser.add_argument("--training", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="y for training n for infer")
    args = parser.parse_args()
    main(args)

