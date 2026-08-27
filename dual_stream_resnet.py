import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES, MODELS, build_backbone
from .rgb_guided_aligner import RGBGuidedAligner


@BACKBONES.register_module()
class DualStreamResNet(BaseModule):

    def __init__(self,
                 rgb_backbone_cfg,
                 normal_backbone_cfg,
                 rectify_cfg,
                 fusion_cfg,
                 init_cfg=None):
        super(DualStreamResNet, self).__init__(init_cfg)

        self.rgb_backbone = build_backbone(rgb_backbone_cfg)
        self.normal_backbone = build_backbone(normal_backbone_cfg)

        rectify_cfg_copy = rectify_cfg.copy()
        rectify_type = rectify_cfg_copy.pop('type')
        in_channels_list = rectify_cfg_copy.pop('in_channels_list')

        self.align_modules = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            RGBGuidedAligner(in_channels_list[2]),
            RGBGuidedAligner(in_channels_list[3]),
        ])

        self.rectify_stages = nn.ModuleList()
        for in_channels in in_channels_list:
            self.rectify_stages.append(
                MODELS.build(dict(
                    type=rectify_type,
                    in_channels=in_channels,
                    **rectify_cfg_copy
                ))
            )

        fusion_cfg_copy = fusion_cfg.copy()
        fusion_type = fusion_cfg_copy.pop('type')
        in_channels_list_ffm = fusion_cfg_copy.pop('in_channels_list')
        num_heads_list = fusion_cfg_copy.pop('num_heads_list')

        self.fusion_stages = nn.ModuleList()
        for i, in_channels in enumerate(in_channels_list_ffm):
            self.fusion_stages.append(
                MODELS.build(dict(
                    type=fusion_type,
                    in_channels=in_channels,
                    num_heads=num_heads_list[i],
                    **fusion_cfg_copy
                ))
            )

        self.align_gate = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(in_channels_list[2], in_channels_list[2] // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels_list[2] // 2, 1, 1),
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels_list[3], in_channels_list[3] // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels_list[3] // 2, 1, 1),
                nn.Sigmoid()
            )
        ])

        num_stages = len(in_channels_list)
        assert (
                len(self.align_modules)
                == len(self.rectify_stages)
                == len(self.fusion_stages)
                == num_stages
        )

    def init_weights(self):
        self.rgb_backbone.init_weights()
        self.normal_backbone.init_weights()

        for align_mod in self.align_modules:
            if hasattr(align_mod, 'init_weights'):
                align_mod.init_weights()

        for r_stage in self.rectify_stages:
            r_stage.init_weights()
        for f_stage in self.fusion_stages:
            f_stage.init_weights()

    def forward(self, img, normal_map):
        x_rgb = self.rgb_backbone(img)

        x_normal = list(self.normal_backbone(normal_map))

        fused_outs = []

        for i in range(len(x_rgb)):
            if i >= 2:
                base_aligned = self.align_modules[i](x_rgb[i], x_normal[i])
                g = self.align_gate[i](x_rgb[i])
                scale = self.align_modules[i].align_scale
                x_normal[i] = x_normal[i] + scale * g * base_aligned

            rgb_rectified, normal_rectified = self.rectify_stages[i](
                x_rgb[i], x_normal[i]
            )

            fused = self.fusion_stages[i](
                rgb_rectified, normal_rectified
            )

            fused_outs.append(fused)

        return tuple(fused_outs)
