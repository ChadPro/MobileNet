# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

from nets import mobile_net_224_original
from nets import mobile_net_300_original
from nets import mobile_net_224_v1
from nets import mobile_net_224_v2
from nets import mobile_net_300_v1
from nets import mobile_net_300_v2


net_map = {
    'mobile_net_224_original' : mobile_net_224_original,
    'mobile_net_300_original' : mobile_net_300_original,
    'mobile_net_224_v2' : mobile_net_224_v1,
    'mobile_net_224_v3' : mobile_net_224_v2,
    'mobile_net_300_v1' : mobile_net_300_v1,
    'mobile_net_300_v2' : mobile_net_300_v2
    }


def get_network(name):
    if name not in net_map:
        raise ValueError('Name of net unknown %s' % name)
    return net_map[name]