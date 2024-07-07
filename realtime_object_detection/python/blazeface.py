from dataclasses import dataclass, field

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
    classes: list = field(default_factory=lambda: ['face'])
    image_size: int = 128
    detection_threshold: float = 0.5
    blazeface_channels: int = 32
    focal_loss: bool = False
    model_path: str = 'weights/blazeface.pt'
    augmentation: dict = None


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
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

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class FinalBlazeBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(FinalBlazeBlock, self).__init__()
        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
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

    def forward(self, x):
        h = F.pad(x, (0, 2, 0, 2), "constant", 0)

        return self.act(self.convs(h))


class BlazeFace(nn.Module):

    def __init__(self, back_model=False):
        super(BlazeFace, self).__init__()

        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 4
        self.score_clipping_thresh = 100.0
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
        self.min_suppression_threshold = 0.3

        self._define_layers()

    def _define_layers(self):
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

    def forward(self, x):
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)

        b = x.shape[0]  # batch size, needed for reshaping later

        if self.back_model:
            x = self.backbone(x)  # (b, 16, 16, 96)
            h = self.final(x)  # (b, 8, 8, 96)
        else:
            x = self.backbone1(x)  # (b, 88, 16, 16)
            h = self.backbone2(x)  # (b, 96, 8, 8)


        c1 = self.classifier_8(x)  # (b, 2, 16, 16)
        c1 = c1.permute(0, 2, 3, 1)  # (b, 16, 16, 2)
        c1 = c1.reshape(b, -1, 3)  # (b, 512, 1)

        c2 = self.classifier_16(h)  # (b, 6, 8, 8)
        c2 = c2.permute(0, 2, 3, 1)  # (b, 8, 8, 6)
        c2 = c2.reshape(b, -1, 3)  # (b, 384, 1)

        c = torch.cat((c1, c2), dim=1)  # (b, 896, 1)

        r1 = self.regressor_8(x)  # (b, 32, 16, 16)
        r1 = r1.permute(0, 2, 3, 1)  # (b, 16, 16, 32)
        r1 = r1.reshape(b, -1, 4)  # (b, 512, 16)

        r2 = self.regressor_16(h)  # (b, 96, 8, 8)
        r2 = r2.permute(0, 2, 3, 1)  # (b, 8, 8, 96)
        r2 = r2.reshape(b, -1, 4)  # (b, 384, 16)

        r = torch.cat((r1, r2), dim=1)  # (b, 896, 16)
        return torch.cat([r, c], dim=2)
        #return [r, c]

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.classifier_8.weight.device

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_anchors(self, path):
        self.anchors = torch.tensor(np.load(path), dtype=torch.float32, device=self._device())
        self.dbox_list = self.anchors
        assert (self.anchors.ndimension() == 2)
        assert (self.anchors.shape[0] == self.num_anchors)
        assert (self.anchors.shape[1] == 4)

