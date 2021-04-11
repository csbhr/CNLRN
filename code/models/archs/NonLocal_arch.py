import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn

    def forward(self, x, b, c):
        bt, _, h, w = x.size()
        t = bt // b
        d_k = c // len(self.patchsize)
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1),
                                                      torch.chunk(_key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1)):
            out_w, out_h = w // width, h // height

            query = query.view(b, t, d_k, out_h, height, out_w, width)
            query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(b, t * out_h * out_w, d_k * height * width)
            key = key.view(b, t, d_k, out_h, height, out_w, width)
            key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(b, t * out_h * out_w, d_k * height * width)
            value = value.view(b, t, d_k, out_h, height, out_w, width)
            value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(b, t * out_h * out_w, d_k * height * width)

            y, _ = self.attention(query, key, value)

            y = y.view(b, t, out_h, out_w, d_k, height, width)
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)

        output = torch.cat(output, 1)
        x = self.output_linear(output)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class NonLocalBlock(nn.Module):
    def __init__(self, patchsize, n_feat=128):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=n_feat)
        self.feed_forward = FeedForward(n_feat)

    def forward(self, input_dict):
        x, b, c = input_dict['x'], input_dict['b'], input_dict['c']
        x = x + self.attention(x, b, c)
        x = x + self.feed_forward(x)
        return {'x': x, 'b': b, 'c': c}
