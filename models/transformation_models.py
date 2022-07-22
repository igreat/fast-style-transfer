import torch.nn as nn
from torchvision import transforms


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        # experiment with different padding
        self.block = nn.Sequential(
            nn.Conv2d(filters, filters, 3),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3),
            nn.BatchNorm2d(filters),
        )

    def forward(self, input):
        x = self.block(input)
        _, _, h, w = x.shape
        cropped_input = transforms.CenterCrop((h, w))(input)
        return x + cropped_input


# defining the transformation model
class TransformationModel(nn.Module):
    def __init__(self):
        super(TransformationModel, self).__init__()
        # add batch normalization between everything
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(40),
            nn.Conv2d(3, 32, 9, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3),
        )

    def forward(self, input):
        return self.layers(input)
