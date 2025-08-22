"""
Utilities for NoDiffTransformer family (cleaned).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def interpolate(inputs,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in inputs.shape[2:])
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
    return F.interpolate(inputs, size, scale_factor, mode, align_corners)


def convs_no_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )


class AttentionDecoder(nn.Module):

    def __init__(self, h, w, in_channels, n_heads=4, attn_drop=0., drop=0., mlp_ratio=4., qkv_bias=True):
        super().__init__()
        assert in_channels % n_heads == 0

        self.h = h
        self.w = w
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.head_dim = int(in_channels / n_heads)
        self.scale = self.head_dim ** -0.5

        # combination layer
        self.q = nn.Linear(in_channels, in_channels, bias=qkv_bias)
        self.kv = nn.Linear(2 * in_channels, 2 * in_channels, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop = nn.Dropout(drop)
        self.proj = nn.Linear(in_channels, in_channels)

    def forward(self, src, tmp):
        batch_size = src.shape[0]

        kv_input = torch.cat((src, tmp), dim=2) # (batch_size, hw, 2 * in_channels)
        q = self.q(src).reshape(batch_size, self.h * self.w, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (bs, n_heads, hw, head_dim)
        kv = self.kv(kv_input).reshape(batch_size, self.h * self.w, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        k_t = k.transpose(-2, -1) # (batch_size, h_heads, head_dim, hw)
        dp = (q @ k_t) * self.scale # (batch_size, n_heads, hw, hw)
        attn = self.attn_drop(dp.softmax(-1))
        weighted_avg = attn @ v # (batch_size, n_heads, hw, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2).flatten(2) # (batch_size, hw, out_dim)

        x = self.proj(weighted_avg)
        x = self.drop(x)
        return x


class TripletAttentionDecoder(nn.Module):

    def __init__(self, h, w, in_channels, n_heads=4, attn_drop=0., drop=0., mlp_ratio=4., qkv_bias=True):
        super().__init__()
        assert in_channels % n_heads == 0

        self.h = h
        self.w = w
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.head_dim = int(in_channels / n_heads)
        self.scale = self.head_dim ** -0.5

        # combination layer
        self.q = nn.Linear(in_channels, in_channels, bias=qkv_bias)
        self.kv = nn.Linear(3 * in_channels, 2 * in_channels, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop = nn.Dropout(drop)
        self.proj = nn.Linear(in_channels, in_channels)

    def forward(self, src, tmp, diff):
        batch_size = src.shape[0]

        kv_input = torch.cat((src, tmp, diff), dim=2) # (batch_size, hw, 3 * in_channels)
        q = self.q(src).reshape(batch_size, self.h * self.w, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # (bs, n_heads, hw, head_dim)
        kv = self.kv(kv_input).reshape(batch_size, self.h * self.w, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        k_t = k.transpose(-2, -1) # (batch_size, h_heads, head_dim, hw)
        dp = (q @ k_t) * self.scale # (batch_size, n_heads, hw, hw)
        attn = self.attn_drop(dp.softmax(-1))
        weighted_avg = attn @ v # (batch_size, n_heads, hw, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2).flatten(2) # (batch_size, hw, out_dim)

        x = self.proj(weighted_avg)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def pos_meshgrid(nx, ny):
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y)

    mesh = torch.zeros((nx, ny, 2))
    mesh[:,:,0] = X 
    mesh[:,:,1] = Y
    return mesh


def convs(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )


class AttentionDecoderBlock(nn.Module):
    """
    Metric tensor that maps two image representations to distance vector
    We add extra positional information along each channel
    """

    def __init__(self, h, w, in_channels, pos_dim=16, n_heads=4, attn_drop=0., drop=0., mlp_ratio=4., qkv_bias=True):
        super().__init__()
        assert (in_channels + pos_dim) % n_heads == 0

        self.h = h
        self.w = w
        self.in_channels = in_channels
        self.pos_dim = pos_dim
        self.n_heads = n_heads
        self.out_dim = in_channels + pos_dim
        self.head_dim = int(self.out_dim / n_heads)
        self.scale = self.head_dim ** -0.5

        # pos encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w, pos_dim))

        # self-attention for search image
        self.src_norm_1 = nn.LayerNorm(self.out_dim)
        self.src_attn = Attention(self.out_dim, num_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.src_norm_2 = nn.LayerNorm(self.out_dim)
        mlp_hidden_dim = int(self.out_dim * mlp_ratio)
        self.src_mlp = ModifiedMLP(in_features=self.out_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

        # combination layer
        self.comb_src_norm_1 = nn.LayerNorm(self.out_dim)
        self.comb_tmp_norm_1 = nn.LayerNorm(self.out_dim)
        self.comb_attn = AttentionDecoder(h, w, self.out_dim, n_heads=n_heads, attn_drop=attn_drop, drop=drop, qkv_bias=qkv_bias)
        mlp_hidden_dim = int(self.out_dim * mlp_ratio)
        self.comb_mlp = ModifiedMLP(in_features=self.out_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        self.comb_norm_2 = nn.LayerNorm(self.out_dim)


    # assumes inputs are (batch_size, hw, in_channels)
    def forward(self, src, tmp):
        batch_size = src.shape[0]

        src = torch.cat((src, self.pos_embed.expand(batch_size, self.h * self.w, self.pos_dim)), dim=2)
        tmp = torch.cat((tmp, self.pos_embed.expand(batch_size, self.h * self.w, self.pos_dim)), dim=2) # (batch_size, hw, in_channels + pos_dim)

        src = src + self.src_attn(self.src_norm_1(src), self.h, self.w)
        src = src + self.src_mlp(self.src_norm_2(src), self.h, self.w) # (batch_size, hw, in_channels + pos_dim)

        x = src + self.comb_attn(self.comb_src_norm_1(src), self.comb_tmp_norm_1(tmp)) # (batch_size, hw, out_dim)
        x = x + self.comb_mlp(self.comb_norm_2(x), self.h, self.w)
        return x


class DepthwiseConv(nn.Module):
    def __init__(self, dim=768):
        super(DepthwiseConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ImageProjection(nn.Module):

    def __init__(self, image_size, kernel_size=7, stride=4, in_channels=1, embed_dim=768):

        super(ImageProjection, self).__init__()

        self.image_size = image_size
        self.kernel_size = kernel_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # change from (B, C, H, W) to (B, HW, C)
        x = self.norm(x)

        return x, H, W


class ModifiedMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DepthwiseConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RelativePositionAttention(nn.Module):
    def __init__(self, h, w, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, pos_dim=64, device=None):
        super(RelativePositionAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.h = h
        self.w = w

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.pos_dim = pos_dim

        self.q = nn.Linear(dim + pos_dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim + pos_dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        # make location embedding by mapping (x,y) to higher dimension
        self.alpha = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w, pos_dim))
        with torch.no_grad():
            self.pos = pos_meshgrid(h, w).reshape(h * w, 2).unsqueeze(0)
            pos_ = pos_meshgrid(h, w).reshape(h * w, 2).unsqueeze(0)
            dist = torch.cdist(self.pos, pos_).unsqueeze(0) # (1, 1, hw, hw)

        if device:
            self.pos = self.pos.to(device)
            dist = dist.to(device)
        self.dist = dist
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = torch.cat((x, self.pos_embed.expand(B, N, self.pos_dim)), dim=-1) # (batch_size, hw, C + pos_embed)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # (batch_size, n_heads, hw, head_dim)

        # concat positional information

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) # (batch_size, n_heads, hw, hw)
        attn = torch.exp(self.alpha * self.dist) * attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TripletAttentionDecoderBlock(nn.Module):
    """
    Metric tensor that maps three image representations to distance vector
    We add extra positional information along each channel
    """

    def __init__(self, h, w, in_channels, pos_dim=16, n_heads=4, attn_drop=0., drop=0., mlp_ratio=4., qkv_bias=True):
        super().__init__()
        assert (in_channels + pos_dim) % n_heads == 0

        self.h = h
        self.w = w
        self.in_channels = in_channels
        self.pos_dim = pos_dim
        self.n_heads = n_heads
        self.out_dim = in_channels + pos_dim
        self.head_dim = int(self.out_dim / n_heads)
        self.scale = self.head_dim ** -0.5

        # pos encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w, pos_dim))

        # self-attention for search image
        self.src_norm_1 = nn.LayerNorm(self.out_dim)
        self.src_attn = Attention(self.out_dim, num_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.src_norm_2 = nn.LayerNorm(self.out_dim)
        mlp_hidden_dim = int(self.out_dim * mlp_ratio)
        self.src_mlp = ModifiedMLP(in_features=self.out_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

        # combination layer
        self.comb_src_norm_1 = nn.LayerNorm(self.out_dim)
        self.comb_tmp_norm_1 = nn.LayerNorm(self.out_dim)
        self.comb_diff_norm_1 = nn.LayerNorm(self.out_dim)
        self.comb_attn = TripletAttentionDecoder(h, w, self.out_dim, n_heads=n_heads, attn_drop=attn_drop, drop=drop, qkv_bias=qkv_bias)
        mlp_hidden_dim = int(self.out_dim * mlp_ratio)
        self.comb_mlp = ModifiedMLP(in_features=self.out_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        self.comb_norm_2 = nn.LayerNorm(self.out_dim)


    # assumes inputs are (batch_size, hw, in_channels)
    def forward(self, src, tmp, diff):
        batch_size = src.shape[0]

        src = torch.cat((src, self.pos_embed.expand(batch_size, self.h * self.w, self.pos_dim)), dim=2)
        tmp = torch.cat((tmp, self.pos_embed.expand(batch_size, self.h * self.w, self.pos_dim)), dim=2) # (batch_size, hw, in_channels + pos_dim)
        diff = torch.cat((diff, self.pos_embed.expand(batch_size, self.h * self.w, self.pos_dim)), dim=2)

        src = src + self.src_attn(self.src_norm_1(src), self.h, self.w)
        src = src + self.src_mlp(self.src_norm_2(src), self.h, self.w) # (batch_size, hw, in_channels + pos_dim)

        x = src + self.comb_attn(self.comb_src_norm_1(src), self.comb_tmp_norm_1(tmp), self.comb_diff_norm_1(diff)) # (batch_size, hw, out_dim)
        x = x + self.comb_mlp(self.comb_norm_2(x), self.h, self.w)
        return x

__all__ = ['Attention', 'AttentionDecoder', 'AttentionDecoderBlock', 'DepthwiseConv', 'ImageProjection', 'ModifiedMLP', 'RelativePositionAttention', 'TripletAttentionDecoder', 'TripletAttentionDecoderBlock', 'convs', 'convs_no_relu', 'interpolate', 'pos_meshgrid']
