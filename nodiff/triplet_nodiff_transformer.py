"""
Model definition for TripletNoDiffTransformer (cleaned).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_transformer_utils import ImageProjection, RelativePositionAttention, ModifiedMLP, interpolate, convs_no_relu, convs, AttentionDecoderBlock, TripletAttentionDecoderBlock
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class ExchangeBlock(nn.Module):

    def __init__(self, p_swap):
        super().__init__()

        self.p_swap = p_swap

    def forward(self, x1, x2, x3):
        b, c, h, w = x1.shape
        exchange_mask_1 = (torch.arange(c) % self.p_swap) == 0
        exchange_mask_2 = (torch.arange(c) % (self.p_swap * 2)) == 0
        exchange_mask_12 = (exchange_mask_1 & (~exchange_mask_2)).unsqueeze(0).expand((b, -1))
        exchange_mask_13 = exchange_mask_2.unsqueeze(0).expand((b, -1))
        exchange_mask_1 = exchange_mask_1.unsqueeze(0).expand((b, -1))

        out_x1, out_x2, out_x3 = torch.zeros_like(x1), torch.zeros_like(x2), torch.zeros_like(x3)
        out_x1[~exchange_mask_1, ...] = x1[~exchange_mask_1, ...]
        out_x2[~exchange_mask_1, ...] = x2[~exchange_mask_1, ...]
        out_x3[~exchange_mask_1, ...] = x3[~exchange_mask_1, ...]

        # three way exchange
        out_x1[exchange_mask_12, ...] = x2[exchange_mask_12, ...]
        out_x1[exchange_mask_13, ...] = x3[exchange_mask_13, ...]
        out_x2[exchange_mask_12, ...] = x1[exchange_mask_12, ...]
        out_x2[exchange_mask_13, ...] = x3[exchange_mask_13, ...]
        out_x3[exchange_mask_12, ...] = x1[exchange_mask_12, ...]
        out_x3[exchange_mask_13, ...] = x2[exchange_mask_13, ...]
        return out_x1, out_x2, out_x3


class Decoder(nn.Module):

    def __init__(self, in_channels=[64, 128, 256, 512],
                 embedding_dim=64, pos_dim=16, output_nc=2, hs=[], ws=[], attn_drop=0., drop=0.):
        super(Decoder, self).__init__()

        self.output_nc = output_nc

        self.hs = hs
        self.ws = ws

        #MLP
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        n = len(in_channels)
        #c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        # linear maps
        self.attn_decoders = []
        self.intermediates = []
        self.conv_outs = []
        for i in range(n):
            self.attn_decoders.append(TripletAttentionDecoderBlock(hs[i], ws[i], in_channels[i], pos_dim=pos_dim))
            self.intermediates.append(convs_no_relu(in_channels=in_channels[i]+pos_dim, out_channels=self.embedding_dim))
            self.conv_outs.append(convs(in_channels=self.embedding_dim, out_channels=self.output_nc))

        self.attn_decoders = nn.ModuleList(self.attn_decoders)
        self.intermediates = nn.ModuleList(self.intermediates)
        self.conv_outs = nn.ModuleList(self.conv_outs)

        self.lin_1 = nn.Linear(2 * sum([h * w for h, w in zip(hs, ws)]), 256)
        self.lin_2 = nn.Linear(256, 1)

        
    def forward(self, x_1, x_2, x_3):
        n, _, h, w = x_1[-1].shape

        x_1 = [x.flatten(2).transpose(1,2) for x in x_1]
        x_2 = [x.flatten(2).transpose(1,2) for x in x_2]
        x_3 = [x.flatten(2).transpose(1,2) for x in x_3]
        outputs = []

        for i in range(len(x_1)):
            out = self.attn_decoders[i](x_1[i], x_2[i], x_3[i]).permute(0,2,1).reshape(n, -1, self.hs[i], self.ws[i])
            out = self.intermediates[i](out)
            out = self.conv_outs[i](out)
            outputs.append(out)

        outputs = [torch.flatten(x, start_dim=1) for x in outputs]
        out = torch.cat(outputs, dim=1)
        out = torch.relu(self.lin_1(out))
        out = self.lin_2(out).squeeze(1)
        return out


class EncoderTransformer(nn.Module):

    def __init__(self, img_size, kernel_size=3, depths=[3, 3, 6, 18], embed_dims=[32, 64, 128, 256],
                 num_heads=[2,2,4,8], drop_path_rate=0., attn_drop_rate=0., drop_rate=0., qkv_bias=True, qk_scale=None, device=None):
        super(EncoderTransformer, self).__init__()
        assert len(depths) == len(embed_dims)
        assert len(depths) == len(num_heads)

        # save important variables
        self.img_size = img_size
        self.depths = depths
        self.embed_dims = [1] + embed_dims
        self.n = len(depths)
        
        kernel_sizes = [7] + [3 for i in range(self.n - 1)]
        strides = [2 for i in range(int(self.n / 2))] + [1 for i in range(self.n - int(self.n / 2))]



        # embedders for projecting input image to higher channel dimension
        self.img_projs = nn.ModuleList([
            ImageProjection(self.img_size, kernel_size=kernel_sizes[i], stride=strides[i],
                          in_channels=self.embed_dims[i], embed_dim=self.embed_dims[i+1]) for i in range(self.n)
        ])

        h, w = img_size
        self.hs, self.ws = [], []
        for i in range(self.n):
            self.hs.append(math.floor((h + 2 * (kernel_sizes[i] // 2) - kernel_sizes[i]) / strides[i] + 1))
            self.ws.append(math.floor((w + 2 * (kernel_sizes[i] // 2) - kernel_sizes[i]) / strides[i] + 1))
            h = self.hs[-1]
            w = self.ws[-1]

        # set up blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.norms = []
        self.blocks = []
        for i in range(self.n):
            block = nn.ModuleList([
                EncoderTransformerBlock(self.hs[i], self.ws[i], dim=self.embed_dims[i+1], num_heads=num_heads[i], drop_path=dpr[cur + j],
                                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, device=device)
                for j in range(depths[i]) 
            ])
            cur += depths[i]
            self.norms.append(nn.LayerNorm(self.embed_dims[i+1]))
            self.blocks.append(block)

        self.blocks = nn.ModuleList(self.blocks)
        self.norms = nn.ModuleList(self.norms)

        self.exchanges = [ExchangeBlock(2) for _ in range(self.n)]

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



    def forward(self, src, temp, diff):
        b, c, h, w = src.shape
        src_outs = []
        temp_outs = []
        diff_outs = []

        x1 = src
        x2 = temp
        x3 = diff
        for i in range(self.n):
            x1, H1, W1 = self.img_projs[i](x1)
            x2, H2, W2 = self.img_projs[i](x2)
            x3, H3, W3 = self.img_projs[i](x3)

            for block in self.blocks[i]:
                x1 = block(x1, H1, W1)
                x2 = block(x2, H2, W2)
                x3 = block(x3, H3, W3)
            x1 = self.norms[i](x1)
            x2 = self.norms[i](x2)
            x3 = self.norms[i](x3)
            x1 = x1.reshape(b, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
            x2 = x2.reshape(b, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
            x3 = x3.reshape(b, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
            x1, x2, x3 = self.exchanges[i](x1, x2, x3)
            src_outs.append(x1)
            temp_outs.append(x2)
            diff_outs.append(x3)

        return src_outs, temp_outs, diff_outs


class EncoderTransformerBlock(nn.Module):

    def __init__(self, h, w, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., drop_path=0., attn_drop=0., mlp_ratio=4, device=None):

        super(EncoderTransformerBlock, self).__init__()

        self.norm_1 = nn.LayerNorm(dim)
        self.attn = RelativePositionAttention(
                h, w,
                dim, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop,
                device=device
            )

        self.norm_2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ModifiedMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

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
        x = x + self.attn(self.norm_1(x), H, W)
        x = x + self.mlp(self.norm_2(x), H, W)
        return x


class TripletNoDiffTransformer(nn.Module):

    def __init__(self, img_size, in_channels=1, embed_dim=256, device=None, depths=[3, 3, 3, 3, 3, 3],
                 num_heads=[2, 2, 4, 8, 8, 8], embed_dims=[64, 128, 256, 512, 512, 512], drop_rate = 0.2, attn_drop = 0.2):

        super().__init__()
        assert len(num_heads) == len(depths)

        self.img_size = img_size
        self.embed_dims = embed_dims
        self.depths = depths
        self.num_heads = num_heads
        self.embedding_dim = embed_dim
        self.drop_rate = drop_rate
        self.attn_drop = attn_drop

        n = len(self.depths)
        self.enc = EncoderTransformer(img_size, embed_dims=self.embed_dims, depths=self.depths, attn_drop_rate=self.attn_drop, 
                                      drop_rate=self.drop_rate, num_heads=self.num_heads, device=device)
        self.hs = self.enc.hs
        self.ws = self.enc.ws
        self.dec = Decoder(in_channels=self.embed_dims, hs=self.hs, ws=self.ws)



    def forward(self, src, temp, diff):
        fx1, fx2, fx3 = self.enc(src, temp, diff)
        diff = self.dec(fx1, fx2, fx3)
        return diff

__all__ = ['TripletNoDiffTransformer']
