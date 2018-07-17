from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import imagenet_300
from datasets import data_imagenet

datasets_map = {
    'imagenet_300' : imagenet_300,
    'imagenet_224' : data_imagenet
}

def get_dataset(name):
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return datasets_map[name]