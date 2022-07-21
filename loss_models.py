from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn
from utils import StyleLoss, ContentLoss, TVLoss


class VGG19Loss(nn.Module):
    def __init__(
        self,
        content_img,
        style_img,
        content_weight=1,
        style_weight=1e4,
        tv_weight=1e-4,
        content_layers=["relu4_2"],
        style_layers=["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"],
        pooling="max",
        device="cpu",
    ):

        super(VGG19Loss, self).__init__()
        features = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
        features.requires_grad_(False)

        self.content_losses = []
        self.style_losses = []

        self.tv_loss = TVLoss(tv_weight)
        self.layers = nn.Sequential()

        pool_cnt, relu_count, conv_count = 1, 1, 1
        for i in range(len(features)):
            x = features[i]
            if isinstance(x, nn.Conv2d):
                name = f"conv{pool_cnt}_{conv_count}"
                conv_count += 1
            elif isinstance(x, nn.ReLU):
                name = f"relu{pool_cnt}_{relu_count}"
                relu_count += 1
            else:
                name = f"pool{pool_cnt}"
                if pooling == "avg":
                    x = nn.AvgPool2d(2, 2)

                relu_count = 1
                conv_count = 1
                pool_cnt += 1

            self.layers.add_module(name, x)

            style_img = x(style_img)
            content_img = x(content_img)

            if name in style_layers:
                loss_module = StyleLoss(style_img, style_weight)
                self.layers.add_module(f"{name}_style_loss", loss_module)
                self.style_losses.append(loss_module)
                style_layers.remove(name)
            elif name in content_layers:
                loss_module = ContentLoss(content_img, content_weight)
                self.layers.add_module(f"{name}_content_loss", loss_module)
                self.content_losses.append(loss_module)
                content_layers.remove(name)

            # making sure it is cut off at the last loss layer to avoid unnecesarry computations
            if len(style_layers) == 0 and len(content_layers) == 0:
                break

    def forward(self, input):
        x = self.tv_loss(input)
        x = self.layers(x)

        # consider making this just return the total loss
        return (
            [content.loss for content in self.content_losses],
            [style.loss for style in self.style_losses],
            self.tv_loss.loss,
        )
