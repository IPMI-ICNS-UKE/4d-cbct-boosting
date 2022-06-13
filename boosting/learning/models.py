import torch
import torch.nn as nn

from boosting.learning.blocks import ConvInstanceNormReLU2D
from boosting.learning.blocks import ResidualDenseBlock2D


class ResidualDenseNet2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            growth_rate: int = 32,
            n_blocks: int = 2,
            n_block_layers: int = 4,
            convolution_block: nn.Module = ConvInstanceNormReLU2D,
            local_feature_fusion_channels: int = 32,
            alpha: float = 0.9,
            pre_block_channels: int = 32,
            post_block_channels: int = 32
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.growth_rate = growth_rate
        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.convolution_block = convolution_block
        self.local_feature_fusion_channels = local_feature_fusion_channels
        self.alpha = alpha

        self.pre_block_channels = pre_block_channels
        self.post_block_channels = post_block_channels

        self.pre_blocks = nn.Sequential(
            self.convolution_block(
                in_channels=self.in_channels,
                out_channels=self.pre_block_channels,
                kernel_size=(3, 3),
                padding='same'
            ),
            self.convolution_block(
                in_channels=self.pre_block_channels,
                out_channels=self.pre_block_channels,
                kernel_size=(3, 3),
                padding='same'
            )
        )

        for i_block in range(self.n_blocks):
            self.add_module(
                f'residual_dense_block_{i_block}',
                ResidualDenseBlock2D(
                    in_channels=self.pre_block_channels if i_block == 0 else self.local_feature_fusion_channels,
                    out_channels=self.local_feature_fusion_channels,
                    growth_rate=self.growth_rate,
                    n_layers=self.n_block_layers
                )
            )

        self.global_feature_fuse = self.convolution_block(
            in_channels=self.pre_block_channels + self.n_blocks * self.local_feature_fusion_channels,
            out_channels=self.post_block_channels,
            kernel_size=(1, 1)
        )

        self.post_blocks = nn.Sequential(
            self.convolution_block(
                in_channels=self.post_block_channels,
                out_channels=self.post_block_channels,
                kernel_size=(3, 3),
                padding='same'
            ),
            nn.Conv2d(
                in_channels=self.post_block_channels,
                out_channels=self.out_channels,
                kernel_size=(3, 3),
                padding='same'
            )
            # no activation function here, i.e. linear activation
        )

    def forward(self, x):

        x_out = self.pre_blocks(x)

        block_outputs = [x_out]
        for i_block in range(self.n_blocks):
            residual_dense_block = self.get_submodule(
                f'residual_dense_block_{i_block}')

            x_out = residual_dense_block(x_out)
            block_outputs.append(x_out)

        x_out = self.global_feature_fuse(
            torch.cat(block_outputs, dim=1)
        )

        x_out = self.post_blocks(x_out)

        # skip connection, scaled with alpha
        x_out = self.alpha * x + (1.0 - self.alpha) * x_out

        return x_out


