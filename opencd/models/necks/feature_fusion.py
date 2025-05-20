# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
from opencd.registry import MODELS
from opencd.models.utils.se_layer import SELayer_v2 as SELayer
from opencd.models.necks.tiny_fpn import TinyFPN


@MODELS.register_module()
class FeatureFusionNeck(BaseModule):
    """Feature Fusion Neck.

    Args:
        policy (str): The operation to fuse features. candidates 
            are `concat`, `sum`, `diff` and `Lp_distance`.
        in_channels (Sequence(int)): Input channels.
        channels (int): Channels after modules, before conv_seg.
        out_indices (tuple[int]): Output from which layer.
    """

    def __init__(self,
                 policy,
                 in_channels=None,
                 channels=None,
                 out_indices=(0, 1, 2, 3),
                 multilevel_in_channels=[1536, 1536, 1536, 1536]):
        super().__init__()
        self.policy = policy
        self.in_channels = in_channels
        self.channels = channels
        self.out_indices = out_indices
        self.multilevel_in_channels = multilevel_in_channels
        self.minus_conv = nn.Sequential(ConvModule(
            in_channels=self.multilevel_in_channels[0],
            out_channels=256,
            kernel_size=1),
            ConvModule(
                in_channels=self.multilevel_in_channels[1],
                out_channels=256,
                kernel_size=1),
            ConvModule(
                in_channels=self.multilevel_in_channels[2],
                out_channels=256,
                kernel_size=1),
            ConvModule(
                in_channels=self.multilevel_in_channels[3],
                out_channels=256,
                kernel_size=1)
        )
        self.channel_att = nn.Sequential(SELayer(768, 256),
                                         SELayer(768, 256),
                                         SELayer(768, 256),
                                         SELayer(768, 256))
        self.channel_att = self.channel_att.to('cuda')

    @staticmethod
    def fusion(x1, x2, policy):
        """Specify the form of feature fusion"""

        _fusion_policies = ['concat', 'sum', 'diff', 'abs_diff']
        assert policy in _fusion_policies, 'The fusion policies {} are ' \
                                           'supported'.format(_fusion_policies)

        if policy == 'concat':
            x = torch.cat([x1, x2], dim=1)
        elif policy == 'sum':
            x = x1 + x2
        elif policy == 'diff':
            x = x2 - x1
        elif policy == 'abs_diff':
            x = torch.abs(x1 - x2)

        return x

    def forward(self, x1, x2):
        """Forward function."""

        assert len(x1) == len(x2), "The features x1 and x2 from the" \
                                   "backbone should be of equal length"
        outs = []

        if self.policy == 'cosine':
            x_orig = [torch.cat([x1[i], x2[i]], dim=1) for i in range(len(x1))]
            x_minus = [self.minus_conv[i](torch.abs(x1[i] - x2[i])) for i in range(len(x1))]
            x_diff = [F.sigmoid(1 - torch.cosine_similarity(x1[i], x2[i], dim=1)).unsqueeze(1) for i in
                      range(len(x1))]
            f = TinyFPN().eval()
            f = f.to('cuda')
            x = f.forward(x_orig)

            x = [torch.cat([x[i] * x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]

            x = [self.channel_att[i](x[i]) for i in range(len(x))]

            outs = [x[i] for i in self.out_indices]

        else:
            for i in range(len(x1)):
                out = self.fusion(x1[i], x2[i], self.policy)
                outs.append(out)

            outs = [outs[i] for i in self.out_indices]

        return tuple(outs)
