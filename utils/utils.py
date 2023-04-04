import math
import random
from bisect import bisect_right
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
############################################################################################
# Channel Compress
############################################################################################

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

class ChannelCompress(nn.Module):
    def __init__(self, in_ch=1024, out_ch=512):
        """
        reduce the amount of channels to prevent final embeddings overwhelming shallow feature maps
        out_ch could be 512, 256, 128
        """
        super(ChannelCompress, self).__init__()
        num_bottleneck = 600
        add_block = []
        add_block += [nn.Linear(in_ch, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.ReLU()]

        add_block += [nn.Linear(num_bottleneck, 500)]
        add_block += [nn.BatchNorm1d(500)]
        add_block += [nn.ReLU()]
        add_block += [nn.Linear(500, out_ch)]

        # Extra BN layer, need to be removed
        #add_block += [nn.BatchNorm1d(out_ch)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.model = add_block

    def forward(self, x):
        x = self.model(x)
        return x
##################################################################################
# Mutual Information Estimator
##################################################################################
class MIEstimator(nn.Module):
    def __init__(self, size1=2048, size2=512):
        super(MIEstimator, self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.in_ch = size1 + size2
        add_block = []
        add_block += [nn.Linear(self.in_ch, 2048)]
        add_block += [nn.BatchNorm1d(2048)]
        add_block += [nn.ReLU()]
        add_block += [nn.Linear(2048, 512)]
        add_block += [nn.BatchNorm1d(512)]
        add_block += [nn.ReLU()]
        add_block += [nn.Linear(512, 1)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.block = add_block

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2, x1_shuff):
        """
        :param x1: observation
        :param x2: representation
        """
        pos = self.block(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.block(torch.cat([x1_shuff, x2], 1))

        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1