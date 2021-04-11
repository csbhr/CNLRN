import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Gradient_Loss(nn.Module):
    def __init__(self, device):
        super(Gradient_Loss, self).__init__()
        self.sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
        self.sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
        self.sobel_filter_X = torch.from_numpy(self.sobel_filter_X).float().to(device)
        self.sobel_filter_Y = torch.from_numpy(self.sobel_filter_Y).float().to(device)

    def forward(self, output, gt):
        b, c, h, w = output.size()

        output_X_c, output_Y_c = [], []
        gt_X_c, gt_Y_c = [], []
        for i in range(c):
            output_grad_X = F.conv2d(output[:, i:i + 1, :, :], self.sobel_filter_X, bias=None, stride=1, padding=1)
            output_grad_Y = F.conv2d(output[:, i:i + 1, :, :], self.sobel_filter_Y, bias=None, stride=1, padding=1)
            gt_grad_X = F.conv2d(gt[:, i:i + 1, :, :], self.sobel_filter_X, bias=None, stride=1, padding=1)
            gt_grad_Y = F.conv2d(gt[:, i:i + 1, :, :], self.sobel_filter_Y, bias=None, stride=1, padding=1)

            output_X_c.append(output_grad_X)
            output_Y_c.append(output_grad_Y)
            gt_X_c.append(gt_grad_X)
            gt_Y_c.append(gt_grad_Y)

        output_X = torch.cat(output_X_c, dim=1)
        output_Y = torch.cat(output_Y_c, dim=1)
        gt_X = torch.cat(gt_X_c, dim=1)
        gt_Y = torch.cat(gt_Y_c, dim=1)

        grad_loss = torch.mean(torch.abs(output_X - gt_X)) + torch.mean(torch.abs(output_Y - gt_Y))

        return grad_loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss
