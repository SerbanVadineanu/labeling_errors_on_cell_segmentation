import torch
from torch import nn


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        target = target.type(torch.LongTensor)

        target = target.to(predict.get_device())
        loss = torch.nn.functional.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss


class DiceLoss(nn.Module):
    """
    Dice loss done only on the last output mask.
    """
    
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target):
        """Calculate Dice Loss for multiple class output.
        Args:
            predict (torch.Tensor): Model output of shape (N x C x H x W).
            target  (torch.Tensor): Target of shape (N x H x W).

        Returns:
            torch.Tensor: Loss value.
        """

        loss = 0
        
        predict = torch.nn.Softmax2d()(predict)[:, 1, :, :]    # We are only interested in the mask corresponding to the cell
        target = target.to(predict.get_device()).view(*predict.shape)

        intersection = torch.sum(predict * target)

        f1 = (2 * intersection) / (torch.sum(predict) + torch.sum(target))
        loss += (1 - f1)

        return loss
