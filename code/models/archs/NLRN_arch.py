from models.archs.NonLocal_arch import NonLocalBlock
import torch.nn as nn
import math


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()

        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Sub-Group
class SubGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_rcabs, nonlocal_psize):
        super(SubGroup, self).__init__()

        patchsize = [(ps, ps) for ps in nonlocal_psize]
        self.nonlocal_block = NonLocalBlock(patchsize=patchsize, n_feat=n_feat)

        rcab_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale)
            for _ in range(n_rcabs)
        ]
        rcab_body.append(conv(n_feat, n_feat, kernel_size))
        self.rcab_body = nn.Sequential(*rcab_body)

    def forward(self, x):
        b, c, h, w = x.size()
        res = self.nonlocal_block({'x': x, 'b': b, 'c': c})['x']
        res = self.rcab_body(res)
        res += x
        return res


## Non-Local Residual Group (NLRG)
class NLRG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_subgroups, n_rcabs, nonlocal_psize):
        super(NLRG, self).__init__()

        modules_body = [
            SubGroup(conv, n_feat, kernel_size, reduction, act, res_scale, n_rcabs, nonlocal_psize)
            for _ in range(n_subgroups)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN)
class NLRN(nn.Module):
    def __init__(self, n_colors=3, n_nlrgs_body=6, n_nlrgs_up1=2, n_nlrgs_up2=2,
                 n_subgroups=2, n_rcabs=4, n_feats=64, kernel_size=3, reduction=16, res_scale=1,
                 nonlocal_psize=(4, 4, 4, 4), conv=default_conv, act=nn.ReLU(True)):
        super(NLRN, self).__init__()

        # define body module
        modules_body = [
            NLRG(
                conv, n_feats, kernel_size, reduction, act, res_scale, n_subgroups, n_rcabs, nonlocal_psize
            )
            for _ in range(n_nlrgs_body)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # define upsample 1 module
        modules_upsample_1 = [
            NLRG(
                conv, n_feats, kernel_size, reduction, act, res_scale, n_subgroups, n_rcabs, nonlocal_psize
            )
            for _ in range(n_nlrgs_up1)
        ]
        modules_upsample_1.append(conv(n_feats, n_feats, kernel_size))
        self.upsample_body_1 = nn.Sequential(*modules_upsample_1)
        self.upsampler_1 = Upsampler(conv, scale=2, n_feat=n_feats, act=False)

        # define upsample 2 module
        modules_upsample_2 = [
            NLRG(
                conv, n_feats, kernel_size, reduction, act, res_scale, n_subgroups, n_rcabs, nonlocal_psize
            )
            for _ in range(n_nlrgs_up2)
        ]
        modules_upsample_2.append(conv(n_feats, n_feats, kernel_size))
        self.upsample_body_2 = nn.Sequential(*modules_upsample_2)
        self.upsampler_2 = Upsampler(conv, scale=2, n_feat=n_feats, act=False)

        # define tail module
        modules_tail = [
            conv(n_feats, n_feats, kernel_size),
            conv(n_feats, n_colors, kernel_size)
        ]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x_feat):
        x_body = self.body(x_feat)
        x_body += x_feat

        x_up_body_1 = self.upsample_body_1(x_body)
        x_up_body_1 += x_body
        x_up_1 = self.upsampler_1(x_up_body_1)

        x_up_body_2 = self.upsample_body_2(x_up_1)
        x_up_body_2 += x_up_1
        x_up_2 = self.upsampler_2(x_up_body_2)

        sr = self.tail(x_up_2)

        return sr
