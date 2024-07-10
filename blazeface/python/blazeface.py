from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelParameters:
    """Class with all the model parameters"""
    batch_size: int = 256
    lr: float = 0.001
    scheduler_type: str = 'ReduceLROnPlateau'
    lr_scheduler_patience: int = 10
    epochs: int = 100
    classes: List[str] = field(default_factory=lambda: ['face'])
    image_size: int = 128
    detection_threshold: float = 0.5
    blazeface_channels: int = 32
    model_path: str = 'weights/blazeface.pt'
    augmentation: Optional[Dict] = None


class BlazeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        """Initialize the BlazeBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int, optional): Size of the convolution kernel. Defaults to 3.
            stride (int, optional): Stride of the convolution. Defaults to 1.
        """
        super(BlazeBlock, self).__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=in_channels, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BlazeBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the BlazeBlock.
        """
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class FinalBlazeBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        """
        Initializes the FinalBlazeBlock.

        Args:
            channels (int): Number of input and output channels.
            kernel_size (int, optional): Size of the convolution kernel. Defaults to 3.
        """
        super(FinalBlazeBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=kernel_size, stride=2, padding=0,
                      groups=channels, bias=True),
            nn.BatchNorm2d(channels),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(channels),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the FinalBlazeBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the FinalBlazeBlock.
        """
        h = F.pad(x, (0, 2, 0, 2), "constant", 0)

        return self.act(self.convs(h))


class BlazeFace(nn.Module):

    def __init__(self, back_model: bool = False) -> None:
        """Initialize the BlazeFace model.

        Args:
            back_model (bool, optional): Whether to use the back model. Defaults to False.
        """
        super(BlazeFace, self).__init__()

        self.num_anchors = 896
        self.back_model = back_model
        if back_model:
            self.x_scale = 256.0
            self.y_scale = 256.0
            self.h_scale = 256.0
            self.w_scale = 256.0
            self.min_score_thresh = 0.65
        else:
            self.x_scale = 128.0
            self.y_scale = 128.0
            self.h_scale = 128.0
            self.w_scale = 128.0
            self.min_score_thresh = 0.75
        self._define_layers()

    def _define_layers(self) -> None:
        """Define the layers of the BlazeFace model."""
        if self.back_model:
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
                nn.ReLU(inplace=True),

                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24, stride=2),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 48, stride=2),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )
            self.final = FinalBlazeBlock(96)
            self.classifier_8 = nn.Conv2d(96, 6, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 18, 1, bias=True)

            self.regressor_8 = nn.Conv2d(96, 8, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 24, 1, bias=True)
        else:
            self.backbone1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
                nn.ReLU(inplace=True),

                BlazeBlock(24, 24),
                BlazeBlock(24, 28),
                BlazeBlock(28, 32, stride=2),
                BlazeBlock(32, 36),
                BlazeBlock(36, 42),
                BlazeBlock(42, 48, stride=2),
                BlazeBlock(48, 56),
                BlazeBlock(56, 64),
                BlazeBlock(64, 72),
                BlazeBlock(72, 80),
                BlazeBlock(80, 88),
            )

            self.backbone2 = nn.Sequential(
                BlazeBlock(88, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )
            self.classifier_8 = nn.Conv2d(88, 6, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 18, 1, bias=True)

            self.regressor_8 = nn.Conv2d(88, 8, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 24, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BlazeFace model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the BlazeFace model.
        """
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        b = x.shape[0]

        if self.back_model:
            x = self.backbone(x)
            h = self.final(x)
        else:
            x = self.backbone1(x)
            h = self.backbone2(x)


        c1 = self.classifier_8(x)
        c1 = c1.permute(0, 2, 3, 1)
        c1 = c1.reshape(b, -1, 3)
        c2 = self.classifier_16(h)
        c2 = c2.permute(0, 2, 3, 1)
        c2 = c2.reshape(b, -1, 3)
        c = torch.cat((c1, c2), dim=1)

        r1 = self.regressor_8(x)
        r1 = r1.permute(0, 2, 3, 1)
        r1 = r1.reshape(b, -1, 4)
        r2 = self.regressor_16(h)
        r2 = r2.permute(0, 2, 3, 1)
        r2 = r2.reshape(b, -1, 4)

        r = torch.cat((r1, r2), dim=1)
        return torch.cat([r, c], dim=2)

    def _device(self) -> torch.device:
        """Which device (CPU or GPU) is being used by this model

        Returns:
            torch.device: The device being used by this model.
        """
        return self.classifier_8.weight.device

    def load_weights(self, path: str) -> None:
        """Load the weights from a file.

        Args:
            path (str): Path to the weights file.
        """
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_anchors(self, path: str) -> None:
        """Loads the anchors from a file.

        Args:
            path (str): Path to the anchors file.
        """
        self.anchors = torch.tensor(np.load(path), dtype=torch.float32, device=self._device())
        self.dbox_list = self.anchors
        assert (self.anchors.ndimension() == 2)
        assert (self.anchors.shape[0] == self.num_anchors)
        assert (self.anchors.shape[1] == 4)

