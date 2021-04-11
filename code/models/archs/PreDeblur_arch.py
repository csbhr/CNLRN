import torch.nn as nn
from models.archs.SRN_arch import SRN


class PreDeblur(nn.Module):

    def __init__(self, n_colors=3, n_deblur_blocks=20, n_feats=64):
        super(PreDeblur, self).__init__()

        self.DeblurNet = SRN(n_colors=n_colors, n_feats=n_feats, n_body_blocks=n_deblur_blocks, output_image=True)

    def forward(self, x):

        deblur = self.DeblurNet(x)

        return deblur
