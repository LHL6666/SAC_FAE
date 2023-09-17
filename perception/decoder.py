import sys
import torch
import numpy as np
import torch.nn as nn

from timm.models.layers import trunc_normal_
from .tae import PatchEmbed, BasicLayer_up, PatchExpand, FinalPatchExpand_X4
from .common import ResBlock, DeFocusBlock, SpatialAttention


class TransformerDecoder(nn.Module):
    r"""
    Swin Transformer
        Refer to the PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(self, obs_shape, feature_dim=300, num_filters=128, num_layers=2, latent_shape=(256, 96),
                 patch_size=4, depths=(2, 2), num_heads=(3, 6, 12, 24), window_size=8, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False,
                 patch_norm=True):
        super().__init__()

        self.ape = ape
        self.mlp_ratio = mlp_ratio
        self.patch_norm = patch_norm
        self.num_layers = len(depths)
        self.feature_dim = feature_dim
        self.num_filters = num_filters

        assert obs_shape[1] == obs_shape[2], "image height != width"

        # for layer norm
        self.num_channels = int(num_filters * 2 ** (self.num_layers - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(obs_shape=obs_shape, patch_size=patch_size, num_filters=num_filters,
                                      norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.latent_shape = latent_shape
        self._decoder_module = nn.ModuleList()
        self.decoder = nn.Linear(in_features=self.feature_dim, out_features=int(np.prod(self.latent_shape)))

        for i_layer in range(self.num_layers):
            if i_layer == 0:
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                         patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                       dim=int(num_filters * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2,
                                       norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(num_filters=int(num_filters * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                           patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                         depth=depths[(self.num_layers-1-i_layer)],
                                         num_heads=num_heads[(self.num_layers-1-i_layer)], window_size=window_size,
                                         mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                         attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None)

            self._decoder_module.append(layer_up)

        self.up = FinalPatchExpand_X4(input_resolution=(obs_shape[1] // patch_size, obs_shape[2] // patch_size),
                                      dim_scale=4, dim=num_filters)
        self.output = nn.Conv2d(in_channels=num_filters, out_channels=obs_shape[0], kernel_size=(1, 1), bias=False)
        self.decoder_norm = norm_layer(self.num_filters)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Decoder
    def forward_decoder(self, x):
        for inx, layer_up in enumerate(self._decoder_module):
            x = layer_up(x)

        x = self.decoder_norm(x)  # B L C

        batch_size, features_len, _ = x.shape

        assert features_len == self.patches_resolution[0] * self.patches_resolution[1], "input features has wrong size"

        x = self.up(x)
        x = x.view(batch_size, 4 * self.patches_resolution[0], 4 * self.patches_resolution[1], -1)
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        x = self.output(x)

        return x

    def forward(self, x, stop_gradient=True):

        latent_var = self.decoder(x)

        if stop_gradient:
            latent_var = latent_var.detach()

        # print("decoder latent shape: ", latent_var.shape)
        latent_var = latent_var.view(-1, self.latent_shape[0], self.latent_shape[1])

        rec_image = self.forward_decoder(latent_var)

        return rec_image


class PixelDecoder(nn.Module):
    """
    A class of convolutional decoder of pixels observation
    """

    def __init__(self, obs_shape, feature_dim=512, num_filters=32, num_layers=4, latent_shape=(3, 35, 35),
                 batch_vel_acc=None):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.init_height = latent_shape[1]
        self.init_width = latent_shape[2]

        print("latent shape: ", latent_shape)

        self.fc = nn.Linear(
            feature_dim, num_filters * latent_shape[1] * latent_shape[2]
        )

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, (3, 3), stride=(1, 1), output_padding=(0, 0))
            )

        self.deconvs.append(
            nn.ConvTranspose2d(
                num_filters, obs_shape[0], (4, 4), stride=(2, 2), output_padding=(0, 0)
            )
        )

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h

        deconv = h.view(-1, self.num_filters, self.init_height, self.init_width)
        self.outputs['deconv1'] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs['obs'] = obs

        return obs

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param(
                'train_decoder/deconv%s' % (i + 1), self.deconvs[i], step
            )
        L.log_param('train_decoder/fc', self.fc, step)


class FocusDecoder(nn.Module):
    """
    A class of custom convolutional decoder of pixels observation
    """

    def __init__(self, obs_shape, feature_dim=200, num_filters=32, num_layers=4, latent_shape=(32, 16, 30)):

        super(FocusDecoder, self).__init__()

        base_channels = num_filters // 2

        self.deconv_block = nn.Sequential(
            # module 0 => [base_channels * 16, 30, 40]
            nn.ConvTranspose2d(in_channels=base_channels * 4, out_channels=base_channels * 4, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1), output_padding=(0, 0), bias=False),
            nn.BatchNorm2d(base_channels * 4, affine=True),
            nn.SELU(inplace=True),

            # DeFocusBlock(gain=2),

            # module 4 => [num_filters, 60, 80]
            nn.ConvTranspose2d(in_channels=base_channels * 4, out_channels=base_channels * 4, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1), output_padding=(0, 0), bias=False),
            nn.BatchNorm2d(base_channels * 4, affine=True),
            nn.SELU(inplace=True),

            DeFocusBlock(gain=2),

            # module 8 => [base_channels, 120, 160]
            nn.ConvTranspose2d(in_channels=base_channels, out_channels=obs_shape[0], kernel_size=(4, 4),
                               stride=(2, 2), padding=(1, 1), output_padding=(0, 0), bias=True),
        )

        self.latent_shape = latent_shape
        self.decoder_layer = nn.Linear(feature_dim, int(np.prod(latent_shape)))

    def forward(self, h):
        h = torch.relu(self.decoder_layer(h))
        deconv_data = h.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2])
        deconv_data = self.deconv_block(deconv_data)

        # print("decoder output: ", deconv_data.shape)

        deconv_data = torch.tanh(deconv_data)

        return deconv_data


class ResidualDecoder(nn.Module):
    """
    A class of custom convolutional decoder of pixels observation
    """

    def __init__(self, obs_shape, feature_dim=200, num_filters=32, num_layers=4, latent_shape=(32, 16, 30)):

        super(ResidualDecoder, self).__init__()

        self.deconv_block = nn.Sequential(
            ResBlock(in_channels=num_filters, out_channels=num_filters, mid_channels=num_filters),
            nn.BatchNorm2d(num_features=num_filters),

            ResBlock(in_channels=num_filters, out_channels=num_filters, mid_channels=num_filters),

            nn.ConvTranspose2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1), output_padding=(0, 0)),
            nn.BatchNorm2d(num_filters),
            nn.SELU(inplace=True),

            nn.ConvTranspose2d(in_channels=num_filters, out_channels=obs_shape[0], kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1), output_padding=(0, 0))
        )

        self.latent_shape = latent_shape
        self.decoder_layer = nn.Linear(feature_dim, int(np.prod(latent_shape)))

    def forward(self, h):
        h = torch.relu(self.decoder_layer(h))
        deconv_data = h.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2])
        deconv_data = self.deconv_block(deconv_data)

        # print("decoder output: ", deconv_data.shape)

        deconv_data = torch.tanh(deconv_data)

        return deconv_data


class AttentionDecoder(nn.Module):
    """
    A class of mask convolutional decoder for masking unuseful information

    """

    def __init__(self, obs_shape, feature_dim=200, num_filters=32, num_layers=4, latent_shape=(32, 16, 30)):

        super(AttentionDecoder, self).__init__()

        self.deconv_block = nn.Sequential(
            ResBlock(in_channels=num_filters, out_channels=num_filters, mid_channels=num_filters),
            nn.BatchNorm2d(num_features=num_filters),

            nn.ConvTranspose2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1),
                               padding=(0, 0), output_padding=(0, 0)),
            nn.BatchNorm2d(num_filters),
            nn.SELU(inplace=True),

            SpatialAttention(in_channels=num_filters, out_channels=num_filters),
            nn.SELU(inplace=True),

            nn.ConvTranspose2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(2, 2),
                               padding=(0, 0), output_padding=(0, 0)),
            nn.BatchNorm2d(num_filters),
            nn.SELU(inplace=True),

            nn.ConvTranspose2d(in_channels=num_filters, out_channels=obs_shape[0], kernel_size=(4, 4), stride=(2, 2),
                               padding=(0, 0), output_padding=(0, 0))
        )

        self.latent_shape = latent_shape
        self.decoder_layer = nn.Linear(in_features=feature_dim, out_features=int(np.prod(latent_shape)))

    def forward(self, latent_var):
        latent_var = torch.relu(self.decoder_layer(latent_var))
        deconv_data = latent_var.view(-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2])
        deconv_data = self.deconv_block(deconv_data)

        # print("decoder output: ", deconv_data.shape)

        deconv_data = torch.tanh(deconv_data)

        return deconv_data


_AVAILABLE_DECODERS = {
    'pixel': PixelDecoder,
    'focus': FocusDecoder,
    'residual': ResidualDecoder,
    'attention': AttentionDecoder,
    'transformer': TransformerDecoder
    }


def make_decoder(decoder_type, obs_shape, feature_dim, num_filters, num_layers, latent_shape):
    assert decoder_type in _AVAILABLE_DECODERS

    return _AVAILABLE_DECODERS[decoder_type](
        obs_shape, feature_dim, num_filters, num_layers, latent_shape
        )
