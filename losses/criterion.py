import torch.nn as nn
import numpy as np

# def dice_score(pred, target):
#     pred_flat = pred.view(-1)
#     target_flat = target.view(-1)
#
#     intersection = (pred_flat * target_flat).sum()
#     union = pred_flat.sum() + target_flat.sum()
#
#     dice_score = (2.0 * intersection) / (union + intersection)
#
#     return dice_score

def dice_score(pred, target):
    intersection = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()

    dice_score = (2.0 * intersection) / (union + 1e-6)

    return dice_score

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input_flat = input.view(-1)
        target_flat = target.view(-1)

        intersection = (input_flat * target_flat).sum()
        union = input_flat.sum() + target_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        dice_loss = 1.0 - dice_score

        return dice_loss
