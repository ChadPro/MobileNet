# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

from datasets import imagenet_224
from datasets import imagenet_300
from datasets import data_imagenet
from datasets import flowers17_224
from datasets import cifar10_224
from datasets import cifar100_224

datasets_map = {
    'imagenet_224' : imagenet_224,
    'imagenet_300' : imagenet_300,
    'flowers17_224' : flowers17_224,
    'cifar10_224' : cifar10_224,
    'cifar100_224' : cifar100_224
}

def get_dataset(name):
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return datasets_map[name]