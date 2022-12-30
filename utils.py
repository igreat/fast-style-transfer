import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import pil_to_tensor, resize
from PIL import Image
from models import loss_models

class StyleLoss(nn.Module):
    def __init__(self, target_feat, weight, batch_size=4):
        super(StyleLoss, self).__init__()
        self.target_gram = (
            get_gram_matrix(target_feat).detach().repeat(batch_size, 1, 1)
        )
        self.strength = weight
        self.mode = "capture"

    def forward(self, gen_feature):
        if self.mode == "loss":
            gram_matrix = get_gram_matrix(gen_feature)
            self.loss = self.strength * F.mse_loss(gram_matrix, self.target_gram)

        return gen_feature


class ContentLoss(nn.Module):
    def __init__(self, weight):
        super(ContentLoss, self).__init__()
        self.strength = weight
        self.mode = "capture"
        self.loss = 0.0

    def forward(self, gen_feature):
        if self.mode == "capture":
            self.target_feature = gen_feature.detach()
        elif self.mode == "loss":
            self.loss = self.strength * F.mse_loss(gen_feature, self.target_feature)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return gen_feature


def get_gram_matrix(featmaps):
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

def preprocess_batch(images: torch.Tensor, loss_model):
    """
    Preprocess a batch of images for the style transfer model.
    Note that the mean and std are stored inside the loss model.
    """
    img = images.float().div(255)

    mean, std = loss_model.MEAN, loss_model.STD
    img = (images - mean) / std

    return img

def deprocess_batch(images: torch.Tensor, loss_model, device="cpu"):
    """
    De-process a batch of images for the style transfer model.
    Note that the mean and std are stored inside the loss model.
    """
    mean, std = loss_model.MEAN.to(device), loss_model.STD.to(device)
    img = images * std + mean
    img = img.clamp(0, 1)
    return img

def preprocess_image(image: torch.Tensor, loss_model):
    """
    Preprocess an image for the style transfer model.
    Note that the mean and std are stored inside the loss model.
    """
    img = image.unsqueeze(0)
    return preprocess_batch(img, loss_model)

def deprocess_image(image: torch.Tensor, loss_model, device="cpu"):
    """
    De-process an image for the style transfer model.
    Note that the mean and std are stored inside the loss model.
    """
    img = deprocess_batch(image, loss_model, device)
    return img.squeeze(0)


def display_images_in_a_grid(
    images: list[np.ndarray], cols: int = 5, titles: list[str] = None
):
    """Display a list of images in a grid."""
    assert (
        (titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ["Image (%d)" % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(int(np.ceil(n_images / float(cols))), cols, n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def apply_style_grid(model, path_to_image, paths_to_models):
    """
    Produces a grid of images in matplotlib for the outputs of multiple models on the same image.
    I used this to compare multiple checkpoints of the same model.
    """

    img = resize(
        pil_to_tensor((Image.open(path_to_image)).convert("RGB"))
        .unsqueeze(0)
        .float()
        .div(255),
        512,
    )
    transformation_model = model.TransformationModel()

    # code to load pretrained models
    models = []
    for path in paths_to_models:
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        transformation_model.load_state_dict(checkpoint["model_state_dict"])
        models.append(transformation_model.eval())

    mean, std = loss_models.VGG16Loss.MEAN, loss_models.VGG16Loss.STD
    gen_images = []
    for model in models:
        gen_image = model(img)
        gen_image = gen_image * std + mean
        gen_image = gen_image.clamp(0, 1)
        gen_image = gen_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        gen_images.append(gen_image)

    # display images in a grid
    display_images_in_a_grid(gen_images, 4, paths_to_models)
    