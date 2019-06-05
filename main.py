# This code was inspired by https://github.com/wbhu/DnCNN-tensorflow

import argparse
from glob import glob
import tensorflow as tf
import time
from model import dncnn
import os
import numpy as np

CORRUPTED_TRAIN_FILENAMES = 'train/scratch_50/noisy/*.png'
ORIGINAL_TRAIN_FILENAMES = 'train/scratch_50/original/*.png'
CORRUPTED_TEST_FILENAMES = 'test/scratch_50/noisy/*.png'
ORIGINAL_TEST_FILENAMES = 'test/scratch_50/original/*.png'

CHECKPOINT_DIR = './checkpoint'
DENOISED_DIR = '../old_data/denoised'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='# of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
# parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default=CHECKPOINT_DIR, help='models are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./data/denoised', help='denoised sample are saved here')
parser.add_argument('--model', dest='model', default='combined', help ='which model to use: denoise, colorize, or both')
args = parser.parse_args()

def train(model):
    lr = args.lr * np.ones([args.epoch])
    # Why is this here? Make global
    lr[30:] = lr[0] / 10.0

    corrupted_frames = glob(CORRUPTED_TRAIN_FILENAMES)
    corrupted_frames = sorted(corrupted_frames)
    original_frames = glob(ORIGINAL_TRAIN_FILENAMES)
    original_frames = sorted(original_frames)
    print('[*] Training') # with', original_frames, corrupted_frames, args.batch_size, args.ckpt_dir, args.epoch, lr)
    model.train(original_frames, corrupted_frames, batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr)

def test(model):
    corrupted_frames = glob(CORRUPTED_TEST_FILENAMES)
    corrupted_frames = sorted(corrupted_frames)
    original_frames = glob(ORIGINAL_TEST_FILENAMES)
    original_frames = sorted(original_frames)
    start = time.time()
    print('[*] Testing') # with', original_frames, corrupted_frames, args.ckpt_dir)
    model.test(original_frames, corrupted_frames, ckpt_dir=args.ckpt_dir, save_dir=DENOISED_DIR)
    end = time.time()
    print("Elapsed time:", end-start)

def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    with tf.Session() as sess:
        # TODO: change to use same model class for all 3, but different params
        model = dncnn(sess, args.model)
        print('Using model', args.model)
        
        if args.phase == 'train':
            train(model)
        elif args.phase == 'test':
            test(model)
        else:
            print('[!]Unknown phase')
            exit(0)

if __name__ == '__main__':
    tf.app.run()
