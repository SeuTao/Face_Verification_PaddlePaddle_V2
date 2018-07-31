# coding=utf-8
import cv2
import numpy as np
import paddle.v2 as paddle
import random
from multiprocessing import cpu_count
from utils import *

class MyReader:
    def __init__(self, imageSize, center_crop_size = 128):
        self.imageSize = imageSize
        self.center_crop_size = center_crop_size
        self.default_image_size = 250

    def train_mapper(self, sample):
        '''
        map image path to type needed by model input layer for the training set
        '''
        img, label = sample
        sparse_label = [0 for i in range(1036)]
        sparse_label[label - 1] = 1

        def crop_img(img, center_crop_size):
            img = cv2.imread(img, 0)
            if center_crop_size < self.default_image_size:
                side = (self.default_image_size - center_crop_size) / 2
                img = img[side: self.default_image_size - side - 1, side: self.default_image_size - side - 1]
            return img

        img = crop_img(img, self.center_crop_size)

        img = cv2.resize(img, (self.imageSize, self.imageSize))

        return img.flatten().astype('float32'), label, sparse_label

    def test_mapper(self, sample):
        '''
        map image path to type needed by model input layer for the test set
        '''
        img, label, _ = sample
        img = paddle.image.load_image(img)
        img = paddle.image.center_crop(img, 128, is_color=True)

        img = cv2.resize(img, (self.imageSize, self.imageSize))
        return img.flatten().astype('float32'), label

    def train_reader(self, train_list, buffered_size=1024):
        def reader():
            with open(train_list, 'r') as f:
                lines = [line.strip() for line in f]

                random.shuffle(lines)
                for line in lines:
                    line = line.strip().split('\t')
                    img_path = line[0]
                    img_label = line[1]

                    yield img_path, int(img_label)

        return paddle.reader.xmap_readers(self.train_mapper, reader, cpu_count(), buffered_size)

    def test_reader(self, test_list, buffered_size=1024):
        def reader():
            with open(test_list, 'r') as f:
                lines = [line.strip() for line in f]
                for line in lines:
                    img_path, lab = line.strip().split('\t')
                    yield img_path, int(lab)

        return paddle.reader.xmap_readers(self.test_mapper, reader,
                                          cpu_count(), buffered_size)