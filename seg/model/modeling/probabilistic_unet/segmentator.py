import torch.nn as nn
import torch

class DeterministicHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # ★ 这里的 in_channels 必须是 64！★
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ProbabilisticUnet(nn.Module):
    def __init__(self, input_channels=65, num_classes=1, num_filters=64, **kwargs):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.unet = None

        # ★ 这里的 in_channels 也必须是 64！★
        self.head = DeterministicHead(in_channels=64, out_channels=num_classes)

    def forward(self, patch, segm=None, coord_features=None, training=True, sample_cnt=1):
        if coord_features is not None:
            patch = torch.cat([patch, coord_features], 1)

        unet_features = self.unet(patch)
        logits = self.head(unet_features)

        return {'instances': logits}

    def sample(self, testing=False):
        raise NotImplementedError("This is a deterministic network now. Use forward() instead.")

    def elbo(self, segm, **kwargs):
        raise NotImplementedError("ELBO is removed. Use standard BCEWithLogitsLoss outside this class.")
