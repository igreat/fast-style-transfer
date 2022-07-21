import torch.nn as nn
import torch.nn.functional as F
import torch


class StyleLoss(nn.Module):
    def __init__(self, target_gram, weight):
        super(StyleLoss, self).__init__()
        self.target_gram = get_gram_matrix(target_gram).detach()
        self.weight = weight

    def forward(self, gen_feature):
        gram_matrix = get_gram_matrix(gen_feature)
        self.loss = self.weight * F.mse_loss(gram_matrix, self.target_gram)
        return gen_feature


class ContentLoss(nn.Module):
    def __init__(self, target_feature, weight):
        super(ContentLoss, self).__init__()
        self.weight = weight
        self.target_feature = target_feature.detach()

    def forward(self, gen_feature):
        self.loss = self.weight * F.mse_loss(
            gen_feature, self.target_feature, reduction="sum"
        )
        return gen_feature


def get_gram_matrix(featmaps):
    _, c, h, w = featmaps.shape
    featmaps = featmaps.view(c, h * w)
    return (featmaps @ featmaps.T).div(h * w)


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
