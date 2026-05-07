# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from thop import profile

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from Swin_Unet import swin_transformer_unet_skip_expand_decoder_sys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, img_size=128, num_classes=8, zero_head=False, vis=False, window_size=8):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.swin_unet = swin_transformer_unet_skip_expand_decoder_sys.SwinTransformerSys(img_size=img_size,
                                patch_size=4,
                                in_chans=3,
                                num_classes=self.num_classes,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=window_size,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits


if __name__ == '__main__':
    net =SwinUnet(img_size=128, num_classes=1)
    a = torch.randn([8,1,128,128])
    out = net(a)
    flops, params = profile(net, inputs=(a,))
    flop_g = flops / (10 ** 9)
    param_mb = params / (1024 * 1024)  # 转换为MB

    print(f"模型的FLOP数量：{flop_g}G")
    print(f"参数数量: {param_mb} MB")
    print(out.shape)