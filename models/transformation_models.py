import torch.nn as nn
from torchvision import transforms


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        # experiment with different padding
        self.block = nn.Sequential(
            nn.Conv2d(filters, filters, 3),
            nn.InstanceNorm2d(filters, affine=True),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3),
            nn.InstanceNorm2d(filters, affine=True),
        )
        self.batch_norm = nn.InstanceNorm2d(filters, affine=True)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.block(input)
        _, _, h, w = x.shape
        cropped_input = transforms.CenterCrop((h, w))(input)
        x = self.batch_norm(x + cropped_input)
        return self.relu(x)


# defining the transformation model
class TransformationModel(nn.Module):
    def __init__(self):
        super(TransformationModel, self).__init__()
        # add batch normalization between everything
        self.layers = nn.Sequential(
            # downsampling layers
            nn.ReflectionPad2d(40),
            nn.Conv2d(3, 32, 9, padding="same"),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            # residual layers
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            # upsampling using fractional convolutions
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3),
            nn.InstanceNorm2d(3, affine=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, content_img):
        x = self.layers(content_img)
        return self.sigmoid(x).mul(255.0)
