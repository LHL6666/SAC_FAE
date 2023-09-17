import sys
import copy
import torch
import numpy as np
import torch.nn as nn

from timm.models.layers import trunc_normal_
from .tae import PatchEmbed, BasicLayer, PatchMerging
from .common import ResBlock, SpatialAttention


def tie_weights(src, target):
    assert type(src) == type(target)
    target.weight = src.weight
    target.bias = src.bias


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim=1024, num_filters=32, num_layers=4, batch_vel_acc=None):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


class TransformerEncoder(nn.Module):
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

    def __init__(self, obs_shape, feature_dim=300, num_filters=128, num_layers=None, batch_vel_acc=None,
                 patch_size=4, depths=(2, 2), num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4., qkv_bias=True,
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

        if obs_shape[1] % 7 == 0:
            window_size = 7
        elif obs_shape[1] % 8 == 0:
            window_size = 8

        # for layer norm
        self.num_channels = int(num_filters * 2 ** (self.num_layers - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(obs_shape=obs_shape, patch_size=patch_size, num_filters=num_filters,
                                      norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, num_filters))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self._encoder_module = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(num_filters=int(num_filters * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                               patches_resolution[1] // (2 ** i_layer)), depth=depths[i_layer],
                               num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self._encoder_module.append(layer)

        self.latent_shape = ((obs_shape[1] / 8) ** 2, num_filters * 2)
        self.encoder = nn.Linear(in_features=int(np.prod(self.latent_shape)), out_features=self.feature_dim, bias=True)
        self.encoder_norm = norm_layer(self.num_channels)

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

    # Encoder and Bottleneck
    def forward_encoder(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self._encoder_module:
            x = layer(x)

        x = self.encoder_norm(x)  # B L C

        return x

    def forward(self, x, stop_gradient=True):
        latent_var = self.forward_encoder(x)

        # print("latent shape: ", latent_var.shape)

        if stop_gradient:
            latent_var = latent_var.detach()

        latent_var = latent_var.view(latent_var.shape[0], -1)
        latent_var = self.encoder(latent_var)

        return latent_var


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""

    def __init__(self, obs_shape, feature_dim, num_filters=32, num_layers=4, batch_vel_acc=None):
        super().__init__()

        assert len(obs_shape) == 3

        _stride = 2
        _padding = 0
        _kernel_size = 4
        _output_shape = list(copy.deepcopy(obs_shape))

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, kernel_size=(_kernel_size, _kernel_size), stride=(_stride, _stride))]
        )

        _output_shape[1] = (_output_shape[1] - _kernel_size + _padding * 2) // _stride + 1
        _output_shape[2] = (_output_shape[2] - _kernel_size + _padding * 2) // _stride + 1

        _stride = 1
        _kernel_size = 3

        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, kernel_size=(_kernel_size, _kernel_size),
                                        stride=(_stride, _stride)))
            _output_shape[1] = (_output_shape[1] - _kernel_size + _padding * 2) // _stride + 1
            _output_shape[2] = (_output_shape[2] - _kernel_size + _padding * 2) // _stride + 1

        # out_dim = {2: 39, 4: 35, 6: 31}[num_layers]
        print("Convs output shape: ", _output_shape)
        self.fc = nn.Linear(num_filters * _output_shape[1] * _output_shape[2], self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_shape = _output_shape

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        # obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.contiguous().view(conv.size(0), -1)
        return h

    def forward(self, obs, stop_gradient=False):
        h = self.forward_conv(obs)

        if stop_gradient:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        out = torch.tanh(h_norm)
        self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], target=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class FocusConcat(nn.Module):
    def __init__(self):
        super(FocusConcat, self).__init__()

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class FocusEncoder(nn.Module):
    """
    A class of custom encoder
    """

    def __init__(self, obs_shape, feature_dim=300, num_filters=32, num_layers=None, batch_vel_acc=None):
        super(FocusEncoder, self).__init__()

        assert len(obs_shape) == 3, "Encoder input data error"

        base_channels = num_filters // 2
        self.res_block_1 = ResBlock(in_channels=num_filters, out_channels=num_filters, mid_channels=num_filters)

        # pre-process observation features
        self.base_block = nn.Sequential(
            # module 0: [obs_shape[0], 120, 160] => [base_channels, 60, 80]
            nn.Conv2d(in_channels=obs_shape[0], out_channels=base_channels, kernel_size=(4, 4), stride=(2, 2),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(base_channels, affine=True),
            nn.SELU(inplace=True),
        )

        self.conv_block = nn.Sequential(
            FocusConcat(),
            # module 1 => [base_channels * 4, 30, 40]
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(base_channels * 4, affine=True),
            nn.SELU(inplace=True),

            # FocusConcat(),
            # module 5 => [base_channels * 8, 15, 20]
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(base_channels * 4, affine=True),
            nn.SELU(inplace=True),
        )

        self.output_shape = [base_channels * 4, 30, 40]
        self.learnable_modules = [1, 2, 4, 5]
        self.res_learnable_modules = self.res_block_1.learnable_modules

        self.output = {}

        # channels * height * width
        self.encoder_layer = nn.Linear(int(np.prod(self.output_shape)), feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def copy_conv_weights_from(self, source):
        """
        Tie convolutional layers and batch norm layers.
        """

        if self.base_block is not None:
            tie_weights(src=source.base_block[0], target=self.base_block[0])
            tie_weights(src=source.base_block[1], target=self.base_block[1])

        for layer in self.learnable_modules:
            # for res_block module
            if layer >= 0:
                tie_weights(src=source.conv_block[layer], target=self.conv_block[layer])
            else:
                # two different res block
                if layer == -1:
                    for i in self.res_learnable_modules:
                        tie_weights(src=source.res_block_1.sub_modules[i], target=self.res_block_1.sub_modules[i])
                elif layer == -2:
                    for i in self.res_learnable_modules:
                        tie_weights(src=source.res_block_2.sub_modules[i], target=self.res_block_2.sub_modules[i])
                elif layer == -3:
                    for i in self.res_learnable_modules:
                        tie_weights(src=source.res_block_3.sub_modules[i], target=self.res_block_3.sub_modules[i])

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs, stop_gradient=False):

        # print("encoder input shape: ", obs.shape)

        base_output = self.base_block(obs)
        conv_output = self.conv_block(base_output)

        conv_output = conv_output.contiguous().view(conv_output.size(0), -1)

        # print("conv output: ", conv_output.shape)
        # print("base output: ", base_output.shape)

        if stop_gradient:
            base_output = base_output.detach()
            conv_output = conv_output.detach()

        self.output['base_block'] = base_output
        self.output['conv_block'] = conv_output

        encoder_output = self.encoder_layer(conv_output)
        encoder_output = torch.tanh(self.layer_norm(encoder_output))

        self.output['encoder_output'] = encoder_output

        # print("encoder latent shape: ", self.output['latent_var'].shape)

        return encoder_output


class ResidualEncoder(nn.Module):
    """
    A class of convolutional residual encoder of pixels observation
    """

    def __init__(self, obs_shape, feature_dim=300, num_filters=32, num_layers=None, batch_vel_acc=None):
        super(ResidualEncoder, self).__init__()

        assert len(obs_shape) == 3, "Encoder input data error"

        self.res_block = ResBlock(in_channels=num_filters, out_channels=num_filters, mid_channels=num_filters, bn=False)
        self.res_block_2 = ResBlock(in_channels=num_filters, out_channels=num_filters, mid_channels=num_filters,
                                    bn=False)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=obs_shape[0], out_channels=num_filters, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(num_filters, affine=True),
            nn.SELU(inplace=True),

            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(num_filters, affine=True),
            nn.SELU(inplace=True),

            self.res_block,
            nn.BatchNorm2d(num_features=num_filters, affine=True),

            self.res_block_2,
            nn.BatchNorm2d(num_features=num_filters, affine=True),
        )

        self.output_shape = [num_filters, 18, 32]
        self.learnable_modules = [0, 1, 3, 4, -1, 7, -2, 9]
        self.res_learnable_modules = self.res_block.learnable_modules

        # channels * height * width
        self.encoder_layer = nn.Linear(int(np.prod(self.output_shape)), feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def copy_conv_weights_from(self, source):
        """
        Tie convolutional layers
        """
        for layer in self.learnable_modules:
            # for res_block module
            if layer >= 0:
                tie_weights(src=source.conv_block[layer], target=self.conv_block[layer])
            else:
                # two different res block
                if layer == -1:
                    for i in self.res_learnable_modules:
                        tie_weights(src=source.res_block.sub_modules[i], target=self.res_block.sub_modules[i])
                elif layer == -2:
                    for i in self.res_learnable_modules:
                        tie_weights(src=source.res_block_2.sub_modules[i], target=self.res_block_2.sub_modules[i])

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs, stop_gradient=False):

        # print("encoder input shape: ", obs.shape)

        conv_output = self.conv_block(obs)

        # print("conv output: ", conv_output.shape)

        conv_output = conv_output.contiguous().view(conv_output.size(0), -1)

        if stop_gradient:
            conv_output = conv_output.detach()

        encoder_output = self.encoder_layer(conv_output)
        encoder_output = self.layer_norm(encoder_output)

        return encoder_output


class AttentionEncoder(nn.Module):
    """
    A class of convolutional encoder of pixels observation
    """

    def __init__(self, obs_shape, feature_dim=300, num_filters=32, num_layers=None, batch_vel_acc=None):
        super(AttentionEncoder, self).__init__()

        assert len(obs_shape) == 3, "Encoder input data error"

        self.res_block = ResBlock(in_channels=num_filters, out_channels=num_filters, mid_channels=num_filters, bn=False)
        self.attn_block = SpatialAttention(in_channels=num_filters, out_channels=num_filters)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=obs_shape[0], out_channels=num_filters, kernel_size=(4, 4), stride=(2, 2), padding=0),
            nn.BatchNorm2d(num_filters, affine=True),
            nn.SELU(inplace=True),

            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(2, 2),
                      padding=0),
            nn.BatchNorm2d(num_filters, affine=True),
            nn.SELU(inplace=True),

            self.attn_block,
            nn.SELU(inplace=True),

            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.BatchNorm2d(num_filters, affine=True),
            nn.SELU(inplace=True),

            self.res_block,
            nn.BatchNorm2d(num_features=num_filters, affine=True),
        )

        self.output_shape = [num_filters, 15, 29]
        self.res_learnable_modules = self.res_block.learnable_modules
        self.attn_learnable_modules = self.attn_block.learnable_modules
        self.learnable_modules = (0, 1, 3, 4, -1, 8, 9, -2, 12)

        # channels * height * width
        self.encoder_layer = nn.Linear(int(np.prod(self.output_shape)), feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

        # print("Encoder: ", self.conv_block)

    def copy_conv_weights_from(self, source):
        """
        Tie convolutional layers
        """
        for layer in self.learnable_modules:
            # for res_block module
            if layer >= 0:
                tie_weights(src=source.conv_block[layer], target=self.conv_block[layer])
            else:
                if layer == -1:
                    for i in self.attn_learnable_modules:
                        # res block lies in the end of the module
                        tie_weights(src=source.attn_block.sub_modules[i], target=self.attn_block.sub_modules[i])
                elif layer == -2:
                    for i in self.res_learnable_modules:
                        tie_weights(src=source.res_block.sub_modules[i], target=self.res_block.sub_modules[i])

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs, stop_gradient=False):

        # print("encoder input shape: ", obs.shape)

        conv_output = self.conv_block(obs)
        # print("conv output: ", conv_output.shape)

        conv_output = conv_output.contiguous().view(conv_output.size(0), -1)

        if stop_gradient:
            conv_output = conv_output.detach()

        encoder_output = self.encoder_layer(conv_output)
        encoder_output = self.layer_norm(encoder_output)

        return encoder_output


_AVAILABLE_ENCODERS = {
    'pixel': PixelEncoder,
    'focus': FocusEncoder,
    'residual': ResidualEncoder,
    'attention': AttentionEncoder,
    'transformer': TransformerEncoder,
}


def make_encoder(encoder_type, obs_shape, feature_dim, num_filters, num_layers):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](obs_shape, feature_dim, num_filters, num_layers)
