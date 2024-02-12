import torch
import torch.nn.functional as F


class SquareThresholdMSELoss:
    def __init__(self, threshold):
        self.threshold = threshold
        self.current_shape = None
        self.mask = None

    def get_mask(self, b, e, device):
        if self.current_shape != (b, e):
            self.current_shape = (b, e)
            self.mask = torch.zeros((b * e, b * e), device=device)
            for i in range(b):
                self.mask[i * e:(i + 1) * e, i * e:(i + 1) * e] = 1
        return self.mask

    def __call__(self, input):
        b, e, *_ = input.shape
        input = input.view(b * e, -1)
        input = torch.where(input < self.threshold, input, 1.0)
        mse_error = (input.unsqueeze(0) - input.unsqueeze(1)).pow(2).mean(-1)
        mask = self.get_mask(b, e, device=input.device)
        return (mse_error * mask).sum() / mask.sum()


class NoCudnnCTCLoss(torch.nn.CTCLoss):
    def __init__(self, *args, **kwargs):
        super(NoCudnnCTCLoss, self).__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        torch.backends.cudnn.enabled = False
        loss = super(NoCudnnCTCLoss, self).forward(*args, **kwargs)
        torch.backends.cudnn.enabled = True
        return loss


class AdversarialHingeLoss:
    def discriminator(self, dis_fake, dis_real):
        loss_real = torch.mean(F.relu(1. - dis_real))
        loss_fake = torch.mean(F.relu(1. + dis_fake))
        return loss_real, loss_fake

    def generator(self, dis_fake):
        loss_fake = -torch.mean(dis_fake)
        return loss_fake
    

class MaxMSELoss:
    def __call__(self, real, fake):
        res = F.mse_loss(real, fake, reduction='none')
        return res.mean(-1).max(1).values.mean()
