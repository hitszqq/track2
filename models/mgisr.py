# -*- coding: utf-8 -*-

import torch
from torch import nn
from functools import partial
from torch.nn import functional as F
from models.irmamba import SS2D, CAB
from typing import  Callable
from models.xformer import TransformerBlock
from models.common import Scale, ConvBNReLU2D, PatchEmbed, PatchUnEmbed, UpSampler


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=3):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=kernel_size, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)

        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))

        self.conv_blk = CAB(hidden_dim, is_light_sr, compress_ratio=2, squeeze_factor=4)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

        self.patch_embed = PatchEmbed(in_chans=hidden_dim, embed_dim=hidden_dim, norm_layer=nn.LayerNorm)
        self.patch_unembed = PatchUnEmbed(in_chans=hidden_dim, embed_dim=hidden_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x * self.skip_scale + self.self_attention(self.ln_1(x))

        conv_x = self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous())
        x = x * self.skip_scale2 + conv_x.permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2)


def nearest_multiple_of_16(n):
    if n % 16 == 0:
        return n
    else:
        lower_multiple = (n // 16) * 16
        upper_multiple = lower_multiple + 16

        if (n - lower_multiple) < (upper_multiple - n):
            return lower_multiple
        else:
            return upper_multiple


class HybridBlock(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.5, window_size=8, overlap_ratio=0.25, num_channel_heads=1,
                 num_spatial_heads=6, spatial_dim_head=16, ffn_expansion_factor=2.66, LayerNorm_type='WithBias',
                 bias=False, d_state=16, expand=2, deps=2):
        super(HybridBlock, self).__init__()
        self.dim = dim
        self.scale = Scale(1)

        self.global_channels = nearest_multiple_of_16(int(global_ratio * dim))
        if self.global_channels + int(local_ratio * dim) > dim:
            self.local_channels = dim - self.global_channels
        else:
            self.local_channels = int(local_ratio * dim)
        self.identity_channels = self.dim - self.global_channels - self.local_channels
        if self.local_channels != 0:
            # self.local_op = nn.Identity()
            self.local_op = nn.Sequential(*[
                TransformerBlock(
                    dim=self.local_channels, window_size=window_size, overlap_ratio=overlap_ratio,
                    num_channel_heads=num_channel_heads, num_spatial_heads=num_spatial_heads,
                    spatial_dim_head=spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor,
                    LayerNorm_type=LayerNorm_type, bias=bias,
                ) for _ in range(deps)
            ])
        else:
            self.local_op = nn.Identity()
        if self.global_channels != 0:
            self.global_op = nn.Sequential(*[
                VSSBlock(hidden_dim=self.global_channels, d_state=d_state, expand=expand) for _ in range(deps)
            ])
        else:
            self.global_op = nn.Identity()

        self.proj = nn.Sequential(*[NAFBlock(dim) for _ in range(deps)])

    def forward(self, x):  # x (B,C,H,W)
        x1, x2, x3 = torch.split(x, [self.global_channels, self.local_channels, self.identity_channels], dim=1)

        x1 = self.global_op(x1) + x1
        x2 = self.local_op(x2) + x2
        return self.proj(torch.cat([x1, x2, x3], dim=1))
        # return x + self.proj(torch.cat([x1, x2, x3], dim=1))


class Head(nn.Module):
    def __init__(self, embed_dim):
        super(Head, self).__init__()
        self.rgb = nn.Sequential(
            ConvBNReLU2D(1, out_channels=embed_dim, kernel_size=7, padding=3),
            NAFBlock(c=embed_dim),
            ConvBNReLU2D(embed_dim, out_channels=embed_dim, kernel_size=3, act='PReLU', padding=1),
        )

        self.nir = nn.Sequential(
            ConvBNReLU2D(1, out_channels=embed_dim, kernel_size=7, padding=3),
            NAFBlock(c=embed_dim),
            ConvBNReLU2D(embed_dim, out_channels=embed_dim, kernel_size=3, act='PReLU', padding=1),
        )
        self.fuse = nn.Sequential(
            ConvBNReLU2D(embed_dim * 2, out_channels=embed_dim, kernel_size=3, padding=1),
            NAFBlock(c=embed_dim),
            ConvBNReLU2D(embed_dim, out_channels=embed_dim, kernel_size=3, act='PReLU', padding=1),
        )

    def forward(self, low_IR, guided):
        out = torch.cat((self.rgb(guided), self.nir(low_IR)), dim=1)
        return self.fuse(out)


class MGISR(torch.nn.Module):
    def __init__(self, args):
        super(MGISR, self).__init__()
        embed_dim = 32
        self.head = Head(embed_dim)

        enc_blks = [4, 6, 8, 12]
        middle_blk_num = 10
        dec_blks = [12, 8, 6, 4]

        local_ratios = [0.25, 0.25, 0.25, 0.25]
        global_ratios = [0.5, 0.5, 0.5, 0.5]

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()

        chan = embed_dim
        for num, global_ratio, local_ratio in zip(enc_blks, global_ratios, local_ratios):
            self.encoders.append(
                HybridBlock(dim=chan, global_ratio=global_ratio, local_ratio=local_ratio, deps=num)
            )

            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))

            chan = chan * 2

        self.middle_blks = nn.Sequential(
            HybridBlock(dim=chan, global_ratio=0.5, local_ratio=0.25, deps=middle_blk_num)
        )

        for num, global_ratio, local_ratio in zip(dec_blks, global_ratios[::-1], local_ratios[::-1]):
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            chan = chan // 2

            self.decoders.append(
                HybridBlock(dim=chan, global_ratio=global_ratio, local_ratio=local_ratio, deps=num)
            )

        self.tail = ConvBNReLU2D(embed_dim, 1, kernel_size=3, padding=1)

        self.padder_size = 2 ** (len(self.encoders) + 3)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def forward(self, low_IR, guided):
        x = self.head(low_IR, guided)
        _, _, H, W = x.size()
        x = self.check_image_size(x)

        out_features = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            out_features.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, out_features[::-1]):
            x = up(x) + enc_skip
            x = decoder(x)
        x = self.tail(x)

        return  x[:, :, :H, :W] + low_IR



if __name__ == '__main__':

    class args:
        embed_dim = 32
        upscale = 8


    img_lr = torch.randn(1, 1, 448, 640).cuda(),
    Guide = torch.randn(1, 3, 448, 640).cuda()

    net = MGISR(args).cuda()

    with torch.no_grad():

        print(net(img_lr, Guide).size())