# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
# print("#####"+str(curPath))
root_path = os.path.abspath(os.path.dirname(curPath) + os.path.sep + ".")
sys.path.append(root_path)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import init
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from utils.supression import SelectDropMAX


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) (nW, nhead, npatch, ndim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (nW, nhead, npatch, npatch) npatch: number patch in a window

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows nW: number windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


"""
MODEL:
  TYPE: swin
  NAME: swin_small_patch4_window7_224
  DROP_PATH_RATE: 0.3
  SWIN:
    EMBED_DIM: 96
    DEPTHS: [ 2, 2, 18, 2 ]
    NUM_HEADS: [ 3, 6, 12, 24 ]
    WINDOW_SIZE: 7
"""



class Swin_Supression_join_pcb(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=26,
                 embed_dim=128, depths=[ 2, 2, 18, 2 ], num_heads=[ 4, 8, 16, 32 ],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.5,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False,supression="join", **kwargs):
        super().__init__()

        self.img_size = img_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # num_patches = self.patch_embed.num_patches
        # patches_resolution = self.patch_embed.patches_resolution
        # self.patches_resolution = patches_resolution

        # # absolute position embedding
        # if self.ape:
        #     self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        #     trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head1 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.conv_4_1 = nn.Conv2d(self.num_features,self.num_features//2,kernel_size=1,stride=1,padding=0)
        self.conv_4_2 = nn.Conv2d(self.num_features//2,self.num_features//4,kernel_size=1,stride=1,padding=0)
        self.conv_4_3 = nn.Conv2d(self.num_features//4,self.num_classes,kernel_size=1,stride=1,padding=0)
        # self.conv_3 = nn.Conv2d(512, num_classes, kernel_size = 1, padding = 0)
        

        self.supression = supression
        self.supression_MAX_3 = SelectDropMAX(supression= self.supression,mask_height=1, mask_width=2)
        # self.supression_Random_3 = SelectDrop(mask_height=3,mask_width=6)

        self.supression_MAX_4 = SelectDropMAX(supression= self.supression,mask_height=1, mask_width=1)
        # self.supression_Random_4 = SelectDrop(mask_height=3,mask_width=3)

        self.avgpool_s = nn.AdaptiveAvgPool1d(1)
        self.norm_s = norm_layer(self.num_classes)
        # self.squeeze1 = nn.Conv2d(self.num_features//2, self.num_classes, kernel_size = 1, padding = 0, bias = False)
        self.squeeze1 = nn.Conv2d(self.num_features//2, self.num_features//4, kernel_size = 1, padding = 0, bias = False)
        self.squeeze2 = nn.Conv2d(self.num_features//4, self.num_classes, kernel_size = 1, padding = 0, bias = False)
        

        self.part_num = 2
        self.drop = nn.Dropout(0.5)

        #############  7个分类器
        self.instance0 = nn.Linear(self.num_classes, self.num_classes)
        # init.normal(self.instance0.weight, std=0.001)
        # init.constant(self.instance0.bias, 0)

        self.instance1 = nn.Linear(self.num_classes, self.num_classes)
        # init.normal(self.instance1.weight, std=0.001)
        # init.constant(self.instance1.bias, 0)

        # self.instance2 = nn.Linear(128, self.num_classes)
        # init.normal(self.instance2.weight, std=0.001)
        # init.constant(self.instance2.bias, 0)

        # self.instance3 = nn.Linear(128, self.num_classes)
        # init.normal(self.instance3.weight, std=0.001)
        # init.constant(self.instance3.bias, 0)

        # self.instance4 = nn.Linear(128, self.num_classes)
        # init.normal(self.instance4.weight, std=0.001)
        # init.constant(self.instance4.bias, 0)

        # self.instance5 = nn.Linear(128, self.num_classes)
        # init.normal(self.instance5.weight, std=0.001)
        # init.constant(self.instance5.bias, 0)

        # self.instance6 = nn.Linear(128, self.num_classes)
        # init.normal(self.instance6.weight, std=0.001)
        # init.constant(self.instance6.bias, 0)

        self.instance_4 = nn.Linear(self.num_classes, self.num_classes)
        # init.normal(self.instance_4.weight, std=0.001)
        # init.constant(self.instance_4.bias, 0)

        self.instance_3 = nn.Linear(self.num_classes, self.num_classes)
        # init.normal(self.instance_3.weight, std=0.001)
        # init.constant(self.instance_4.bias, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        R = []
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        # Wh, Ww = 64,64
        
        # if self.ape: #no
        #     x = x + self.absolute_pos_embed
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x) #-->([16, 3136, 128])

        for layer in self.layers: 
            # print("####"+str(Wh)+"#####"+str(Ww))
            x_out, H, W, x, Wh, Ww = layer(x,Wh, Ww) #([16, 3136, 128]),([16, 784, 256]),([16, 196, 512]),([16, 49, 1024])
            R.append(x_out)
            # print("layer##########"+str(x_out.size()))
        feat = self.norm(x)  # B L C
        feat = self.avgpool(feat.transpose(1, 2))  # B C 1
        feat = torch.flatten(feat, 1)
        return R,feat

    def forward(self, x):
        R,feat = self.forward_features(x) #x.size= (16,49,1024)
        ######没有丢失的原特征###########
        output = self.head1(feat)

        if self.training:
            x3 = self.change_size(R[3])
            x2 = self.change_size(R[2])
##############################################################################################################
            # # 对第三层的特征进行分块
            # x2 = self.drop(x2)
            fmap2 = self.squeeze1(x2)
            fmap2 = self.squeeze2(fmap2) #bs,classes,14,14
            parts = fmap2.chunk(2, dim = 2) # bs,classes,7,14  分成2parts
            # print("$$$$$$$$$$$$$$$$$$"+str(parts[0].size()))
            # 对第三层特征进行抑制
            # parts_sup_r = [self.supression_Random_3(p) for p in parts] # 抑制后的bs,128,7,14
            # parts_sup = [torch.tensor(p,dtype = torch.float32).cuda() for p in parts_sup_r]
            parts_sup = [self.supression_MAX_3(p) for p in parts]  #### bs,128,7,14
            
            # reshape
            parts_sup_flatten =[p.flatten(2).permute(0,2,1).cuda() for p in parts_sup]
            parts_sup_norm=[self.norm_s(p) for p in parts_sup_flatten]
            batch_supression_2 = [self.avgpool_s(p.transpose(1, 2)) for p in parts_sup_norm] # B C 1 bs,26,1
            c0 = torch.flatten(batch_supression_2[0], 1)
            c1 = torch.flatten(batch_supression_2[1], 1)
            # c2 = torch.flatten(batch_supression_2[2], 1)
            # c3 = torch.flatten(batch_supression_2[3], 1)
            
            # parts_sup_avg =[F.avg_pool2d(p,kernel_size=(p.size(2),p.size(3)),stride=(p.size(2),p.size(3))) for p in parts_sup]
            # parts_sup_max =[F.max_pool2d(p,kernel_size=(p.size(2),p.size(3)),stride=(p.size(2),p.size(3))) for p in parts_sup] #7个 bs,128,1,1

            # parts_sup_join_0 = parts_sup_avg[0]+parts_sup_max[0]
            # parts_sup_join_1 = parts_sup_avg[1]+parts_sup_max[1]

            # c0 = parts_sup_join_0.contiguous().view(parts_sup_join_0.size(0),-1) #bs,128 
            # c1 = parts_sup_join_1.contiguous().view(parts_sup_join_1.size(0),-1) #bs,128 


            # x_2 = parts_sup[2].contiguous().view(parts_sup[2].size(0),-1) 
            # x_3 = parts_sup[3].contiguous().view(parts_sup[3].size(0),-1) 
            # x_4 = parts_sup[4].contiguous().view(parts_sup[4].size(0),-1) 
            # x_5 = parts_sup[5].contiguous().view(parts_sup[5].size(0),-1) 
            # x_6 = parts_sup[6].contiguous().view(parts_sup[6].size(0),-1)



            # c0 = self.instance0(c0)  #nn.linear 线性层
            # c1 = self.instance1(c1)
            # c2 = self.instance2(x_2)
            # c3 = self.instance3(x_3)
            # c4 = self.instance4(x_4)
            # c5 = self.instance5(x_5)
            # c6 = self.instance6(x_6)

            # parts_sup_rsp_nm = [self.norm_s(p) for p in parts_sup_rsp]  # B L C bs,49,128
            # batch_supression_2 = [self.avgpool_s(p.transpose(1, 2)) for p in parts_sup_rsp_nm]   # B C 1
            # batch_supression_2 = [torch.flatten(p, 1) for p in batch_supression_2] 
            # cat the feats
            # batch_supression_2 = torch.cat(batch_supression_2, dim = -1) #bs,52
            # output_s_2 = self.head2(batch_supression_2)
 #################################################################################################################           
        
            # 对第四层特征做抑制
            # x3 = self.drop(x3)
            x3 = self.conv_4_1(x3)#16,26,7,7
            x3 = self.conv_4_2(x3)
            x3 = self.conv_4_3(x3)
            x3_sup = self.supression_MAX_4(x3) #b,c,h,w

            parts_sup_flatten_3 =x3_sup.flatten(2).permute(0,2,1).cuda() #b,l,c
            parts_sup_norm_3=self.norm_s(parts_sup_flatten_3)
            batch_supression_3 = self.avgpool_s(parts_sup_norm_3.transpose(1, 2))  # B C 1 bs,26,1
            output_s_3 = torch.flatten(batch_supression_3, 1)

            # x3_sup_avg = F.avg_pool2d(x3_sup,kernel_size=(x3_sup.size(2),x3_sup.size(3)),stride=(x3_sup.size(2),x3_sup.size(3)))
            # x3_sup_max = F.max_pool2d(x3_sup,kernel_size=(x3_sup.size(2),x3_sup.size(3)),stride=(x3_sup.size(2),x3_sup.size(3)))
            # x3_sup_join = x3_sup_avg+x3_sup_max

            # output_s_3 = x3_sup_join.contiguous().view(x3_sup_avg.size(0),-1)

            # output_s_3 = self.instance_4(output_s_3)


            # x3_sup_rsp = x3_sup.flatten(2).permute(0,2,1).cuda() #16,49,26
            # x3_sup_rsp_nm = self.norm_s(x3_sup_rsp)  # B L C bs,49,26
            # batch_supression_3 = self.avgpool_s(x3_sup_rsp_nm.transpose(1, 2))  # B C 1 bs,26,1
            # output_s_3 = torch.flatten(batch_supression_3, 1)
            
            return output,output_s_3,(c0,c1)
            # return output,output_s_3,output_s_2
        else:
            return output

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
    def change_size(self,x):
        B, L, C = x.shape
        H,W = int(L**0.5),int(L**0.5)
        x = x.view(B, H, W, C).permute(0,3,1,2)
        return x

if __name__ == '__main__':
    x = torch.rand(16,3,224,224)
    model = Swin_Supression_join_pcb()
    model.cuda()
    print(x.size())
    y = model(x.cuda())
    # print(y.size())
