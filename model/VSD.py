from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import softplus
from utils.utils import ChannelCompress

__all__ = ['ft_net']

##################################################################################
# Initialization function
##################################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

##################################################################################
# Variational Information Bottleneck
##################################################################################
class VIB(nn.Module):
    def __init__(self, in_ch=1024, z_dim=216, num_class=26):
        super(VIB, self).__init__()
        self.in_ch = in_ch
        self.out_ch = z_dim * 2 #512
        self.num_class = num_class
        self.bottleneck = ChannelCompress(in_ch=self.in_ch, out_ch=self.out_ch)
        # classifier of VIB, maybe modified later.
        classifier = []
        classifier += [nn.Linear(self.out_ch, self.out_ch // 2)]
        classifier += [nn.BatchNorm1d(self.out_ch // 2)]
        classifier += [nn.LeakyReLU(0.1)]
        classifier += [nn.Dropout(0.5)]
        classifier += [nn.Linear(self.out_ch // 2, self.num_class)]
        classifier = nn.Sequential(*classifier)
        self.classifier = classifier
        self.classifier.apply(weights_init_classifier)

    def forward(self, v):
        z_given_v = self.bottleneck(v)
        p_y_given_z = self.classifier(z_given_v)
        return p_y_given_z, z_given_v

