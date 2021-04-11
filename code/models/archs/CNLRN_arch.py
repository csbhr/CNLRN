import torch.nn as nn
from models.archs.SRN_arch import SRN
from models.archs.NLRN_arch import NLRN


class CNLRN(nn.Module):

    def __init__(self, n_colors=3, n_deblur_blocks=20, n_nlrgs_body=6, n_nlrgs_up1=2, n_nlrgs_up2=2,
                 n_subgroups=2, n_rcabs=4, n_feats=64, nonlocal_psize=(4, 4, 4, 4), scale=4):
        super(CNLRN, self).__init__()

        assert scale == 4, "only support scale=4."

        self.DeblurNet = SRN(n_colors=n_colors, n_feats=n_feats, n_body_blocks=n_deblur_blocks, output_image=False)

        self.SRNet = NLRN(
            n_colors=n_colors, n_nlrgs_body=n_nlrgs_body, n_nlrgs_up1=n_nlrgs_up1,
            n_nlrgs_up2=n_nlrgs_up2, n_subgroups=n_subgroups, n_rcabs=n_rcabs,
            n_feats=n_feats, nonlocal_psize=nonlocal_psize
        )

    def forward(self, x):

        deblur_feat = self.DeblurNet(x)
        sr = self.SRNet(deblur_feat)

        return sr
