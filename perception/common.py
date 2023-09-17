import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from enum import Enum
from torchvision import models
from torch.autograd import Variable
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """
    Apply orthogonal initialization for Conv2D and Linear layers.

    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)

        if m.bias is not None:
            m.bias.data.fill_(0.0)

        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def xavier_init(m):
    """
    Apply Xavier initialization for Conv2D and Linear layers.

    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)

        if m.weight is not None:
            nn.init.xavier_uniform_(m.weight)

        if m.bias is not None:
            nn.init.xavier_uniform_(m.bias)


def log_normal_density(x, mean, log_std, std):
    """
    returns gaussian density given x on log scale

    """

    variance = std.pow(2)
    # num_env * frames * act_size
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 * np.log(2 * np.pi) - log_std
    # num_env * frames * 1
    log_density = log_density.sum(dim=-1, keepdim=True)

    return log_density


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        self.learnable_modules = [1, 3] if bn is False else [1, 2, 4]

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.SiLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))

        self.sub_modules = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.sub_modules(x)


class FocusBlock(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class DeFocusBlock(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class HSigmoid(nn.Module):

    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            HSigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int = 32, out_channels: int = 32, rate: int = 4):
        super(SpatialAttention, self).__init__()

        self.learnable_modules = [0, 1, 3, 4]

        layers = [
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=(7, 7), padding=3),
            nn.BatchNorm2d(int(in_channels / rate), affine=True),
            nn.ReLU(),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=(7, 7), padding=3),
            nn.BatchNorm2d(out_channels)
        ]

        self.sub_modules = nn.Sequential(*layers)

    def forward(self, x):

        x_spatial_att = self.sub_modules(x).sigmoid()
        out = x * x_spatial_att

        return out


class Vgg19_out(nn.Module):
    def __init__(self, requires_grad=False, input_channel=3, device=torch.device('cuda')):
        super(Vgg19_out, self).__init__()

        # models.VGG19_Weights.DEFAULT
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).to(device).eval()
        vgg_pretrained_features = vgg.features

        if input_channel != 3:
            vgg_pretrained_features[0] = nn.Conv2d(input_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # print("VGG features: ", vgg.features[0])

        self.requires_grad = requires_grad
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        # (3)
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # (3, 7)
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # (7, 12)
        for x in range(9, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # (12, 21)
        for x in range(14, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # (21, 30)
        for x in range(23, 32):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 100.0 * (1 - super(MS_SSIM_Loss, self).forward(img1, img2))


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100.0 * (1 - super(SSIM_Loss, self).forward(img1, img2))


class PerceptualLoss(nn.Module):
    def __init__(self, input_channel: int = 3, block_indexes: tuple = (1, 3, 5),
                 block_weights: tuple = (1.0 / 1.0, 1.0 / 1.0, 1.0), requires_grad=False, device=torch.device('cuda')):

        super(PerceptualLoss, self).__init__()

        self.vgg_network = Vgg19_out(requires_grad=requires_grad, input_channel=input_channel, device=device).to(device)

        self._device = device
        self._criterion = nn.MSELoss()
        self._block_indexes = block_indexes
        self._block_weights = block_weights if len(block_weights) == len(block_indexes) else (1.0, 1.0, 1.0, 1.0, 1.0)

        assert len(block_weights) == len(block_indexes), "the length indexes and weights should match each other!"

    def forward(self, x, y):
        # print("x shape: ", x.shape)

        x_vgg, y_vgg = self.vgg_network(x), self.vgg_network(y)

        loss = 0.0
        for block in self._block_indexes:
            block_loss = self._block_weights[self._block_indexes.index(block)] * self._criterion(x_vgg[block - 1],
                                                                                                 y_vgg[
                                                                                                 block - 1].detach())

            loss += block_loss

            # print(f"perceptual block {block} loss: ", block_loss, " weight: ",
            #       self._block_weights[self._block_indexes.index(block)])

        return loss


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, activation=True):
        super(ConvBlock, self).__init__()

        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=padding,
                              groups=groups, bias=True)

        self.activation = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.conv(x))


class CustomPerceptualLoss(nn.Module):
    """
    Refer to VGG19 Network
    Keep the same number filters to classify the features captured from multiple channels

    """
    def __init__(self, input_channel: int = 3, output_channel: int = 32, block_indexes: tuple = (1, 3, 5),
                 block_weights: tuple = (1.0 / 1.0, 1.0 / 1.0, 1.0), device=torch.device('cuda')):
        super(CustomPerceptualLoss, self).__init__()

        self._device = device
        self._criterion = nn.MSELoss()
        self._block_indexes = block_indexes
        self._block_weights = block_weights if len(block_weights) == len(block_indexes) else (1.0, 1.0, 1.0, 1.0, 1.0)

        assert len(block_weights) == len(block_indexes), "the length indexes and weights should match each other!"

        self.vgg_block_1 = self._make_stage(input_channel, output_channel, num_blocks=2, max_pooling=True).to(
            device).eval()
        self.vgg_block_2 = self._make_stage(output_channel, output_channel, num_blocks=2,
                                            max_pooling=True).to(device).eval()
        self.vgg_block_3 = self._make_stage(output_channel, output_channel, num_blocks=4,
                                            max_pooling=True).to(device).eval()
        self.vgg_block_4 = self._make_stage(output_channel, output_channel, num_blocks=4,
                                            max_pooling=True).to(device).eval()
        self.vgg_block_5 = self._make_stage(output_channel, output_channel, num_blocks=4,
                                            max_pooling=True).to(device).eval()

        self.apply(weight_init)

    @staticmethod
    def _make_stage(in_channels, out_channels, num_blocks, max_pooling):

        layers = [ConvBlock(in_channels, out_channels, kernel_size=3, stride=1)]

        for _ in range(1, num_blocks):
            layers.append(ConvBlock(out_channels, out_channels, kernel_size=3, stride=1))

        if max_pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self._device)
        out1 = self.vgg_block_1(x)
        out2 = self.vgg_block_2(out1)
        out3 = self.vgg_block_3(out2)
        out4 = self.vgg_block_4(out3)
        out5 = self.vgg_block_5(out4)

        return [out1, out2, out3, out4, out5]

    def perceptual_loss(self, x, y):
        x_features = self.forward(x)
        y_features = self.forward(y)

        loss = 0.0
        for block in self._block_indexes:
            block_loss = self._block_weights[self._block_indexes.index(block)] * self._criterion(x_features[block - 1],
                                                                                                 y_features[
                                                                                                 block - 1].detach())
            loss += block_loss
            print(f"perceptual block {block} loss: ", block_loss, " weight: ",
                  self._block_weights[self._block_indexes.index(block)])

        return loss


class SoftDiceLoss(nn.Module):
    """
    Soft dice loss calculation, assumes the `channels_last` format.
    Refer to V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation

    Parameters
        ----------
    epsilon: float
        Used for numerical stability to avoid divide by zero errors

    """

    def __init__(self, epsilon=1e-6):
        super(SoftDiceLoss, self).__init__()

        self._epsilon = epsilon

    def forward(self, y_true, y_pred):
        """
        Soft dice loss function

        Parameters
        ----------
        y_true: tensor
            [batch, channel, H, W]: One hot encoding of ground truth

        y_pred: tensor
            [batch, channel, H, W]: Network output, must sum to 1 over c channel (such as after softmax)

        Returns
        ----------

        """
        axes = tuple(range(2, len(y_pred.shape)))
        numerator = 2. * np.sum(y_pred * y_true, axes)
        denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

        return 1 - np.mean(numerator / (denominator + self._epsilon))
