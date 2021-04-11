import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, padding=0, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, padding=padding, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Conv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=1, padding=0, bias=True, bn=False, act=False):
        super(Conv, self).__init__()
        m = []
        m.append(nn.Conv2d(input_channels, n_feats, kernel_size, stride, padding, bias=bias))
        if bn: m.append(nn.BatchNorm2d(n_feats))
        if act: m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, input):
        return self.body(input)


class Deconv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=2, padding=0, output_padding=0, bias=True,
                 act=False):
        super(Deconv, self).__init__()
        m = []
        m.append(nn.ConvTranspose2d(input_channels, n_feats, kernel_size, stride=stride, padding=padding,
                                    output_padding=output_padding, bias=bias))
        if act: m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, input):
        return self.body(input)


class SRN(nn.Module):
    def __init__(self, n_colors=3, n_feats=64, n_body_blocks=20, kernel_size=5, output_image=True):
        super(SRN, self).__init__()

        self.inBlock = nn.Sequential(
            Conv(n_colors, n_feats, kernel_size, padding=2, act=True),
            ResBlock(Conv, n_feats, kernel_size, padding=2),
            ResBlock(Conv, n_feats, kernel_size, padding=2),
            ResBlock(Conv, n_feats, kernel_size, padding=2)
        )

        # encoder1
        self.encoder_first = nn.Sequential(
            Conv(n_feats, n_feats * 2, kernel_size, padding=2, stride=2, act=True),
            ResBlock(Conv, n_feats * 2, kernel_size, padding=2),
            ResBlock(Conv, n_feats * 2, kernel_size, padding=2),
            ResBlock(Conv, n_feats * 2, kernel_size, padding=2)
        )

        # encoder2
        self.encoder_second = Conv(n_feats * 2, n_feats * 4, kernel_size, padding=2, stride=2, act=True)

        self.body = nn.Sequential(
            *[ResBlock(Conv, n_feats * 4, kernel_size, padding=2) for _ in range(n_body_blocks)]
        )

        # decoder2
        self.decoder_second = Deconv(n_feats * 4, n_feats * 2, kernel_size=3, padding=1, output_padding=1, act=True)

        # decoder1
        self.decoder_first = nn.Sequential(
            ResBlock(Conv, n_feats * 2, kernel_size, padding=2),
            ResBlock(Conv, n_feats * 2, kernel_size, padding=2),
            ResBlock(Conv, n_feats * 2, kernel_size, padding=2),
            Deconv(n_feats * 2, n_feats, kernel_size=3, padding=1, output_padding=1, act=True)
        )

        if output_image:
            self.outBlock = nn.Sequential(
                ResBlock(Conv, n_feats, kernel_size, padding=2),
                ResBlock(Conv, n_feats, kernel_size, padding=2),
                ResBlock(Conv, n_feats, kernel_size, padding=2),
                Conv(n_feats, n_colors, kernel_size, padding=2)
            )
        else:
            self.outBlock = nn.Sequential(
                ResBlock(Conv, n_feats, kernel_size, padding=2),
                ResBlock(Conv, n_feats, kernel_size, padding=2),
                ResBlock(Conv, n_feats, kernel_size, padding=2)
            )

    def forward(self, x):
        x_inblock = self.inBlock(x)
        x_encoder_first = self.encoder_first(x_inblock)
        x_encoder_second = self.encoder_second(x_encoder_first)
        x_body = self.body(x_encoder_second)
        x_decoder_second = self.decoder_second(x_body)
        x_decoder_first = self.decoder_first(x_decoder_second + x_encoder_first)
        x_outBlock = self.outBlock(x_decoder_first + x_inblock)

        return x_outBlock
