import torch.nn as nn
from typing import Optional
from boosting.learning.models import ResidualDenseNet2D
from boosting.learning.blocks import ConvReLU2D

class CBCTBooster:
    def __init__(
            self,
            model: Optional[nn.Module] = None
    ):
        # no model passed, initialize non-trained instance w/ standard params
        if not model:
            self.model = ResidualDenseNet2D(
                in_channels=1,
                out_channels=1,
                growth_rate=16,
                n_blocks=2,
                n_block_layers=4,
                convolution_block=ConvReLU2D,
                local_feature_fusion_channels=32,
                alpha=0.9,
                pre_block_channels=32,
                post_block_channels=32
            )
        else:
            self.model = model



