import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch


class StyleLoss(nn.Module):
    def __init__(self, target_gram, weight):
        super(StyleLoss, self).__init__()
        self.target_gram = get_gram_matrix(target_gram).detach()
        self.weight = weight
        self.mode = "capture"

    def forward(self, gen_feature):
        if self.mode == "loss":
            gram_matrix = get_gram_matrix(gen_feature)
            self.loss = self.weight * F.mse_loss(gram_matrix, self.target_gram)

        return gen_feature


class ContentLoss(nn.Module):
    def __init__(self, weight):
        super(ContentLoss, self).__init__()
        self.weight = weight
        self.mode = "capture"

    def forward(self, gen_feature):
        if self.mode == "capture":
            self.target_feature = gen_feature.detach()
        elif self.mode == "loss":
            self.loss = self.weight * F.mse_loss(gen_feature, self.target_feature)
            self.mode = "loss"

        return gen_feature


def get_gram_matrix(featmaps):
    # make this return applicable for inputs with batch size > 1!
    b, c, h, w = featmaps.shape
    featmaps = featmaps.view(b, c, h * w)
    output = (featmaps @ featmaps.transpose(1, 2)).div(c * h * w)
    return output


# Total variation loss
class TVLoss(nn.Module):
    def __init__(self, weight):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, featmaps):
        self.x_diff = featmaps[:, :, 1:, :] - featmaps[:, :, :-1, :]
        self.y_diff = featmaps[:, :, :, 1:] - featmaps[:, :, :, :-1]
        self.loss = self.weight * (
            torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff))
        )
        return featmaps
