
import torch
import torch.nn as nn

import torchtask
from haishoku.haishoku import Haishoku
import numpy as np
import math
from torchvision import utils as vutils

def func(t):
    if (t > 0.008856):
        return np.power(t, 1/3.0)
    else:
        return 7.787 * t + 16 / 116.0

def rgb2lab(rgb):
    #Conversion Matrix
    matrix = [[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]]

    cie = np.dot(matrix, list(rgb)) / 255

    cie[0] = cie[0] /0.950456
    cie[2] = cie[2] /1.088754

    # Calculate the L
    L = 116 * np.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1]

    # Calculate the a
    a = 500*(func(cie[0]) - func(cie[1]))

    # Calculate the b
    b = 200*(func(cie[1]) - func(cie[2]))

    return [L, a, b]


def add_parser_arguments(parser):
    torchtask.criterion_template.add_parser_arguments(parser)



def harmonizer_loss():
    return HarmonizerLoss


class AbsoluteLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(AbsoluteLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, gt):
        loss = torch.sqrt((pred - gt) ** 2 + self.epsilon)
        return loss

class ColorLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(ColorLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, gt):
        vutils.save_image(pred, './pred.jpg', normalize=True)
        vutils.save_image(gt, './gt.jpg', normalize=True)

        palette_p = Haishoku.getPalette('pred.jpg')
        palette_g = Haishoku.getPalette('gt.jpg')

        weights = []
        labs = []
        loss = 0
        for p in palette_p:
            weights.append(p[0])
            lab = rgb2lab(p[1])
            labs.append(lab)
        i = 0
        for p in palette_g:
            weight = min(weights[i], p[0])
            l_g, a_g, b_g = rgb2lab(p[1])
            l_p, a_p, b_p = labs[i]

            loss = loss + math.sqrt((l_p - l_g)**2 + (a_p - a_g)**2 + (b_p - b_g)**2 + self.epsilon) * weight

        loss = max(loss - 1.5, 0)

        return loss

class HarmonizerLoss(torchtask.criterion_template.TaskCriterion):
    def __init__(self, args):
        super(HarmonizerLoss, self).__init__(args)

        self.l1 = AbsoluteLoss()
        self.l2 = nn.MSELoss(reduction='none')
        self.cl = ColorLoss()

    def forward(self, pred, gt, inp):
        pred_outputs, = pred
        x, mask = inp

        assert len(pred_outputs) == len(gt)

        image_losses = []
        pre = None
        for pred_, gt_ in zip(pred_outputs, gt):
            l1_loss = torch.sum(self.l1(pred_, gt_) * mask, dim=(1, 2, 3)) / (torch.sum(mask, dim=(1, 2, 3)) + 1e-6)
            l2_loss = torch.sum(self.l2(pred_, gt_) * mask, dim=(1, 2, 3)) / (torch.sum(mask, dim=(1, 2, 3)) + 1e-6) * 10
            color_loss = torch.sum(self.cl(pred_, pre) * mask, dim=(1, 2, 3)) / (torch.sum(mask, dim=(1, 2, 3)) + 1e-6) if pre is not None else 0
            loss = 0.4 * (l1_loss + l2_loss) + 0.6 * color_loss
            image_losses.append(loss)
            pre = pred_
        return image_losses
