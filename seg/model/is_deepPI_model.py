import torch.nn as nn
import torch

from seg.utils.serialization import serialize
from seg.model.base import ISModel
from seg.model.modifiers import LRMult
from seg.model.ops import DistMaps
# 注意：这里的 ProbabilisticUnet 是我们在 segmentator.py 里改好的“确定性空壳”
from seg.model.modeling.probabilistic_unet.segmentator import ProbabilisticUnet


class DeepPI(ISModel):
    @serialize
    def __init__(self, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        # 实例化我们改好的确定性基座
        self.feature_extractor = ProbabilisticUnet(input_channels=64 + 1, num_classes=1)
        self.feature_extractor.apply(LRMult(backbone_lr_mult))
        self.dist_maps = DistMaps(norm_radius=20, spatial_scale=1.0,
                                  cpu_mode=False, use_disks=True)

    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features

    # 核心清理：彻底删除了 sample_cnt, un_weight, un_scale 等概率参数
    def backbone_forward(self, image, mask=None, coord_features=None, training=True):
        # 直接调用 feature_extractor 获取确定性的 logits，不再计算 elbo 和 samples
        outputs = self.feature_extractor(patch=image, segm=mask, coord_features=coord_features, training=training)
        return outputs

    # 核心清理：彻底删除了 sample_cnt, un_weight, un_scale 等概率参数
    def forward(self, image, points, mask=None, training=True):
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        click_map = coord_features[:, 1:, :, :]
        coord_features = self.maps_transform(coord_features)

        # 极速单次前向传播
        outputs = self.backbone_forward(image=image, mask=mask, coord_features=coord_features, training=training)

        outputs['click_map'] = click_map

        return outputs
