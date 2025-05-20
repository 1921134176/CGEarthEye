# _*_ coding:utf-8 _*_
# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model.weight_init import xavier_init
from mmengine.model import BaseModule

from opencd.registry import MODELS
import warnings
import torch.nn.functional as F
from .feature_fusion import FeatureFusionNeck


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MultiLevelNeck(nn.Module):
    """MultiLevelNeck.

    A neck structure connect vit backbone and decoder_heads.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        scales (List[float]): Scale factors for each input feature map.
            Default: [0.5, 1, 2, 4]
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 scales=[0.5, 1, 2, 4],
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)
        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.lateral_convs.append(
                ConvModule(
                    in_channel,
                    out_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        for _ in range(self.num_outs):
            self.convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # for len(inputs) not equal to self.num_outs
        if len(inputs) == 1:
            inputs = [inputs[0] for _ in range(self.num_outs)]
        outs = []
        for i in range(self.num_outs):
            x_resize = resize(
                inputs[i], scale_factor=self.scales[i], mode='bilinear')
            outs.append(self.convs[i](x_resize))
        return tuple(outs)


@MODELS.register_module()
class MultiLevelFeatureFusionNeck(BaseModule):
    def __init__(self,
                 multilevel_in_channels,
                 multilevel_out_channels,
                 featurefusion_policy,
                 multilevel_scales=[0.5, 1, 2, 4],
                 multilevel_norm_cfg=None,
                 multilevel_act_cfg=None,
                 featurefusion_in_channels=None,
                 featurefusion_channels=None,
                 featurefusion_out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.multilevel_neck = MultiLevelNeck(multilevel_in_channels,
                                              multilevel_out_channels,
                                              multilevel_scales,
                                              multilevel_norm_cfg,
                                              multilevel_act_cfg,)
        self.featurefusion_neck = FeatureFusionNeck(featurefusion_policy,
                                                    featurefusion_in_channels,
                                                    featurefusion_channels,
                                                    featurefusion_out_indices,
                                                    multilevel_in_channels)

    def forward(self, x1, x2):
        x1 = self.multilevel_neck(x1)
        x2 = self.multilevel_neck(x2)
        return self.featurefusion_neck(x1, x2)

