import torch
import models.archs.CNLRN_arch as CNLRN_arch
import models.archs.PreDeblur_arch as PreDeblur_arch
import models.archs.discriminator_vgg_arch as SRGAN_arch
import models.archs.RRDBNet_arch as RRDBNet_arch
import models.archs.EDVR_arch as EDVR_arch


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'CNLRN':
        netG = CNLRN_arch.CNLRN(
            n_colors=opt_net['n_colors'], n_deblur_blocks=opt_net['n_deblur_blocks'],
            n_nlrgs_body=opt_net['n_nlrgs_body'], n_nlrgs_up1=opt_net['n_nlrgs_up1'],
            n_nlrgs_up2=opt_net['n_nlrgs_up2'], n_subgroups=opt_net['n_subgroups'],
            n_rcabs=opt_net['n_rcabs'], n_feats=opt_net['n_feats'],
            nonlocal_psize=opt_net['nonlocal_psize'], scale=opt_net['scale'])
    elif which_model == 'PreDeblur':
        netG = PreDeblur_arch.PreDeblur(
            n_colors=opt_net['n_colors'], n_deblur_blocks=opt_net['n_deblur_blocks'], n_feats=opt_net['n_feats']
        )
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    # video restoration
    elif which_model == 'EDVR':
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
