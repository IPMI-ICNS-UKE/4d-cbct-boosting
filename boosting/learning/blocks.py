from __future__ import annotations

from typing import Union, Tuple, Type

import torch
import torch.nn as nn


class _ConvNormActivation(nn.Module):
    def __init__(
        self,
        convolution,
        normalization,
        activation,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        padding="same",
    ):
        super().__init__()
        self.convolution = convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        if normalization:
            self.normalization = normalization(out_channels)
        else:
            self.normalization = None
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        if self.normalization:
            x = self.normalization(x)
        x = self.activation(x)

        return x


class ConvInstanceNormReLU2D(_ConvNormActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        padding="same",
    ):
        super().__init__(
            convolution=nn.Conv2d,
            normalization=nn.InstanceNorm2d,
            activation=nn.ReLU,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )


class ConvInstanceNormReLU3D(_ConvNormActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        padding="same",
    ):
        super().__init__(
            convolution=nn.Conv3d,
            normalization=nn.InstanceNorm3d,
            activation=nn.ReLU,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )


class ConvReLU2D(_ConvNormActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        padding="same",
    ):
        super().__init__(
            convolution=nn.Conv2d,
            normalization=None,
            activation=nn.ReLU,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )


class ConvReLU3D(_ConvNormActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        padding="same",
    ):
        super().__init__(
            convolution=nn.Conv3d,
            normalization=None,
            activation=nn.ReLU,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )


class ConvInstanceNormMish2D(_ConvNormActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        padding="same",
    ):
        super().__init__(
            convolution=nn.Conv2d,
            normalization=nn.InstanceNorm2d,
            activation=nn.Mish,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )


class ConvInstanceNormMish3D(_ConvNormActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        padding="same",
    ):
        super().__init__(
            convolution=nn.Conv3d,
            normalization=nn.InstanceNorm3d,
            activation=nn.Mish,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )


class _ResidualDenseBlock(nn.Module):
    def __init__(
        self,
        convolution_block: Type[nn.Module],
        in_channels: int,
        out_channels: int = 32,
        growth_rate: int = 16,
        n_layers: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.growth_rate = growth_rate
        self.n_layers = n_layers

        for i_layer in range(self.n_layers):
            conv = convolution_block(
                in_channels=in_channels,
                out_channels=self.growth_rate,
                kernel_size=3,
                padding="same",
            )
            name = f"conv_block_{i_layer}"

            self.add_module(name, conv)

            in_channels = (i_layer + 1) * self.growth_rate + self.in_channels

        self.local_feature_fusion = convolution_block(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            padding="same",
        )

    def forward(self, x):
        outputs = []
        for i_layer in range(self.n_layers):
            layer = self.get_submodule(f"conv_block_{i_layer}")
            stacked = torch.cat((x, *outputs), dim=1)
            x_out = layer(stacked)
            outputs.append(x_out)

        # print(f'outputs: {[m.shape for m in outputs]}')
        stacked = torch.cat((x, *outputs), dim=1)
        x_out = self.local_feature_fusion(stacked)

        x_out = x + x_out

        return x_out


class ResidualDenseBlock2D(_ResidualDenseBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        growth_rate: int = 16,
        n_layers: int = 4,
        convolution_block: Type[nn.Module] = ConvInstanceNormReLU2D,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            growth_rate=growth_rate,
            n_layers=n_layers,
            convolution_block=convolution_block,
        )


class ResidualDenseBlock3D(_ResidualDenseBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        growth_rate: int = 16,
        n_layers: int = 4,
        convolution_block: Type[nn.Module] = ConvInstanceNormReLU3D,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            growth_rate=growth_rate,
            n_layers=n_layers,
            convolution_block=convolution_block,
        )


class EncoderBlock(nn.Module):
    def __init__(
        self,
        convolution_layer: Type[nn.Module],
        downsampling_layer: Type[nn.Module],
        norm_layer: Type[nn.Module],
        in_channels,
        out_channels,
        n_convolutions: int = 1,
        convolution_kwargs: dict | None = None,
        downsampling_kwargs: dict | None = None,
    ):
        super().__init__()

        if not convolution_kwargs:
            convolution_kwargs = {}
        if not downsampling_kwargs:
            downsampling_kwargs = {}

        self.down = downsampling_layer(**downsampling_kwargs)

        layers = []
        for i_conv in range(n_convolutions):
            layers.append(
                convolution_layer(
                    in_channels=in_channels if i_conv == 0 else out_channels,
                    out_channels=out_channels,
                    **convolution_kwargs,
                )
            )
            if norm_layer:
                layers.append(norm_layer(out_channels))
            layers.append(nn.LeakyReLU(inplace=True))
        self.convs = nn.Sequential(*layers)

    def forward(self, *inputs):
        x = self.down(*inputs)
        return self.convs(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        convolution_layer: Type[nn.Module],
        upsampling_layer: Type[nn.Module],
        norm_layer: Type[nn.Module],
        in_channels,
        out_channels,
        n_convolutions: int = 1,
        convolution_kwargs: dict | None = None,
        upsampling_kwargs: dict | None = None,
    ):
        super().__init__()

        if not convolution_kwargs:
            convolution_kwargs = {}
        if not upsampling_kwargs:
            upsampling_kwargs = {}

        self.up = upsampling_layer(**upsampling_kwargs)

        layers = []
        for i_conv in range(n_convolutions):
            layers.append(
                convolution_layer(
                    in_channels=in_channels if i_conv == 0 else out_channels,
                    out_channels=out_channels,
                    **convolution_kwargs,
                )
            )
            if norm_layer:
                layers.append(norm_layer(out_channels))
            layers.append(nn.LeakyReLU(inplace=True))
        self.convs = nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.convs(x)
        return x
