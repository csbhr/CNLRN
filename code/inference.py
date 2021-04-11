import os
import os.path as osp
import glob
import logging
import argparse
import numpy as np
import cv2
import torch
import utils.util as util
import data.util as data_util
import models.archs.CNLRN_arch as CNLRN_arch


def main(name_flag, input_path, gt_path, model_path, save_path, save_imgs, flip_test):
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    save_path = os.path.join(save_path, name_flag)

    #### model
    model = CNLRN_arch.CNLRN(
        n_colors=3, n_deblur_blocks=20, n_nlrgs_body=6, n_nlrgs_up1=2, n_nlrgs_up2=2,
        n_subgroups=2, n_rcabs=4, n_feats=64, nonlocal_psize=(4, 4, 4, 4), scale=4
    )
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    #### logger
    util.mkdirs(save_path)
    util.setup_logger('base', save_path, name_flag, level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    logger.info('Evaluate: {}'.format(name_flag))
    logger.info('Input images path: {}'.format(input_path))
    logger.info('GT images path: {}'.format(gt_path))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Results save path: {}'.format(save_path))
    logger.info('Flip test: {}'.format(flip_test))
    logger.info('Save images: {}'.format(save_imgs))

    #### Evaluation
    total_psnr_l = []
    total_ssim_l = []

    img_path_l = sorted(glob.glob(osp.join(input_path, '*')))

    #### read LQ and GT images
    imgs_LQ = data_util.read_img_seq(input_path)
    img_GT_l = []
    for img_GT_path in sorted(glob.glob(osp.join(gt_path, '*'))):
        img_GT_l.append(data_util.read_img(None, img_GT_path))

    # process each image
    for img_idx, img_path in enumerate(img_path_l):
        img_name = osp.splitext(osp.basename(img_path))[0]
        imgs_in = imgs_LQ[img_idx:img_idx + 1].to(device)

        if flip_test:
            output = util.flipx8_forward(model, imgs_in)
        else:
            output = util.single_forward(model, imgs_in)

        output = util.tensor2img(output.squeeze(0))

        if save_imgs:
            cv2.imwrite(osp.join(save_path, '{}.png'.format(img_name)), output)

        # calculate PSNR
        output = output / 255.
        GT = np.copy(img_GT_l[img_idx])

        output, GT = util.crop_border([output, GT], crop_border=4)
        crt_psnr = util.calculate_psnr(output * 255, GT * 255)
        crt_ssim = util.ssim(output * 255, GT * 255)
        total_psnr_l.append(crt_psnr)
        total_ssim_l.append(crt_ssim)

        logger.info('{} \tPSNR: {:.3f} \tSSIM: {:.4f}'.format(img_name, crt_psnr, crt_ssim))

    logger.info('################ Final Results ################')
    logger.info('Evaluate: {}'.format(name_flag))
    logger.info('Input images path: {}'.format(input_path))
    logger.info('GT images path: {}'.format(gt_path))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Results save path: {}'.format(save_path))
    logger.info('Flip test: {}'.format(flip_test))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Total Average PSNR: {:.3f} SSIM: {:.4f} for {} images.'.format(
        sum(total_psnr_l) / len(total_psnr_l),
        sum(total_ssim_l) / len(total_ssim_l),
        len(img_path_l))
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNLRN-Inference')
    parser.add_argument('--name_flag', type=str, default='CNLRN_Val300',
                        help='Inference flag name')
    parser.add_argument('--input_path', type=str, default='../dataset/Val300/blur_bicubic_X4',
                        help='Input images path')
    parser.add_argument('--gt_path', type=str, default='../dataset/Val300/sharp',
                        help='GT images path')
    parser.add_argument('--model_path', type=str, default='../pretrain_models/CNLRN.pth',
                        help='Model path')
    parser.add_argument('--save_path', type=str, default='../infer_results',
                        help='Results save path')
    parser.add_argument('--save_imgs', action='store_true', help='save image if true')
    parser.add_argument('--flip_test', action='store_true', help='using self emsemble if true')
    args = parser.parse_args()

    main(
        name_flag=args.name_flag, input_path=args.input_path, gt_path=args.gt_path,
        model_path=args.model_path, save_path=args.save_path,
        save_imgs=args.save_imgs, flip_test=args.flip_test
    )
