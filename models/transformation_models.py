import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Classic residual block, but with instance normalization
    instead of batch normalization, and no relu at the end
    """

    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        # experiment with different padding

        self.conv1 = ReflectConv(filters, filters, 3, stride=1)
        self.instance_norm1 = nn.InstanceNorm2d(filters, affine=True)
        self.conv2 = ReflectConv(filters, filters, 3, stride=1)
        self.instance_norm2 = nn.InstanceNorm2d(filters, affine=True)

    def forward(self, input):
        output = self.conv1(input)
        output = F.relu(self.instance_norm1(output))
        output = self.instance_norm2(self.conv2(output))
        return output + input


# defining the transformation model
class TransformationModel(nn.Module):
    """
    The transformation model is a neural network that takes a
    content image as input and outputs a transformed image.
    """

    def __init__(self):
        super(TransformationModel, self).__init__()
        # add instance norm normalization between everything
        self.layers = nn.Sequential(
            # downsampling layers
            ReflectConv(3, 32, 9, stride=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            ReflectConv(32, 64, 3, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            ReflectConv(64, 128, 3, stride=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            # residual layers
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            # upsampling layers
            UpsampleConv(128, 64, 3, stride=1, scale_factor=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            UpsampleConv(64, 32, 3, stride=1, scale_factor=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            ReflectConv(32, 3, 9, stride=1),
        )

    def forward(self, content_img):
        return self.layers(content_img)


class ReflectConv(nn.Module):
    """
    Reflection padding convolutional layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ReflectConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            padding_mode="reflect",
        )

    def forward(self, x):
        return self.conv(x)


class UpsampleConv(nn.Module):
    """
    Upsampling followed by a convolutional layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, scale_factor):
        super(UpsampleConv, self).__init__()
        self.scale_factor = scale_factor
        self.conv = ReflectConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.conv(x)
