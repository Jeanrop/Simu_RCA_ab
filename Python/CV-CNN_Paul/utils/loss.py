import torch
from torch import nn


def get_loss(type, args=None):
    if type == "dice":
        return DiceLoss(**args)
    elif type =='L2':
        return nn.MSELoss()
    elif type =='rmse':
        return RMSELoss(**args)
    elif type =='smoothL1':
        return nn.SmoothL1Loss()
    else:
        raise NotImplementedError('available: dice, L2, rmse')


class DiceLoss(nn.Module):
    def __init__(self, global_dice=True, as_loss=True, reduce=True):
        super().__init__()
        self.global_dice = global_dice
        self.as_loss = as_loss
        self.reduce = reduce

    def forward(self, input, target):
        smooth = 1.
        target = target.unsqueeze(1)
        intersection = input * target
        # if we compute the global dice then we will some over the batch dim,
        # otherwise no
        if self.global_dice:
            dim = list(range(0, len(input.shape)))
        else:
            dim = list(range(1, len(input.shape)))
        intersection = intersection.sum(dim=dim)
        card_pred = input.sum(dim)
        card_target = target.sum(dim)

        # return 1-dice if it is used as loss
        if self.as_loss:
            res = 1 - ((2. * intersection + smooth) /
                       (card_pred + card_target + smooth))

        else:
            res = ((2. * intersection + smooth) /
                   (card_pred + card_target + smooth))
        # reduce or return the result for each sample of the batch
        if self.reduce:
            return res.mean()
        else:
            return res

class RMSELoss(nn.Module):
    def __init__(self,eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self,x,y):
        return torch.sqrt(self.mse(x,y)+ self.eps)
