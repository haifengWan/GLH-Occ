# Copyright (c) OpenMMLab.
# DUHR: Depth-Uncertainty Guided Height-Aware Recalibration Head
#
# 工程目的：
# 1. 不修改原始 BEVOCCHead2D；
# 2. 新增一个可配置的 Occupancy Head；
# 3. 在 2D BEV feature 到 channel-to-height 解码之前，
#    增加“高度感知 + 深度不确定性”重标定模块。

import math
import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import numpy as np

from mmdet3d.models.builder import HEADS, build_loss
from .bev_occ_head import nusc_class_frequencies, BEVOCCHead2D


class DUHRModule(nn.Module):
    """
    Depth-Uncertainty Guided Height-Aware Recalibration Module.

    输入:
        x: (B, C, H, W), FlashOcc 的 BEV feature

    输出:
        out: (B, C, H, W), 重标定后的 BEV feature

    设计思想:
        1. 将 C 个通道按 Dz 个高度层分组；
        2. 为每个 BEV 网格预测 Dz 个高度权重；
        3. 根据高度分布熵估计 depth/height uncertainty；
        4. 用 uncertainty 进一步调节 height-aware gate；
        5. 用残差形式输出，避免破坏官方预训练权重。

    说明:
        这里的 depth uncertainty 是 BEV-level depth/height uncertainty proxy。
        它不改变 FlashOcc 主干 forward，也不需要从 LSSViewTransformer 额外传 depth_prob。
        后续如果要进一步增强，可以再把真实 depth distribution 接入此模块。
    """

    def __init__(
        self,
        channels=256,
        Dz=16,
        reduction=4,
        use_residual=True,
        zero_init=True,
    ):
        super(DUHRModule, self).__init__()

        assert channels % Dz == 0, (
            f"DUHRModule requires channels % Dz == 0, "
            f"but got channels={channels}, Dz={Dz}"
        )

        self.channels = channels
        self.Dz = Dz
        self.group_channels = channels // Dz
        self.use_residual = use_residual

        mid_channels = max(channels // reduction, 32)

        # 提取 BEV 上下文，用于预测高度层权重。
        self.context_conv = nn.Sequential(
            ConvModule(
                channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            )
        )

        # 高度层 logits: (B, Dz, H, W)
        self.height_logits = nn.Conv2d(mid_channels, Dz, kernel_size=1)

        # 不确定性映射分支:
        # 由 height distribution 的 entropy 得到 uncertainty map，
        # 再映射成 Dz 个高度层偏置。
        self.uncertainty_proj = nn.Sequential(
            nn.Conv2d(1, Dz, kernel_size=1, bias=True),
            nn.Conv2d(Dz, Dz, kernel_size=3, stride=1, padding=1, groups=Dz, bias=True)
        )

        # 高度 gate 的轻量 refinement。
        self.height_refine = nn.Conv2d(
            Dz, Dz, kernel_size=3, stride=1, padding=1, groups=Dz, bias=True
        )

        # 残差缩放因子。
        # zero_init=True 时，模块初始近似 identity，有利于加载官方 FlashOcc M1 权重 fine-tune。
        if zero_init:
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape

        context = self.context_conv(x)

        # height_logits: (B, Dz, H, W)
        height_logits = self.height_logits(context)

        # height_prob 表示每个 BEV 网格上的高度分布。
        # 用 softmax 后的熵作为 depth/height uncertainty proxy。
        height_prob = torch.softmax(height_logits, dim=1)

        # entropy: (B, 1, H, W), 归一化到 [0, 1] 附近。
        entropy = -(height_prob * torch.log(height_prob.clamp_min(1e-6))).sum(
            dim=1, keepdim=True
        ) / math.log(self.Dz)

        # confidence 越大，说明高度/深度越确定。
        confidence = 1.0 - entropy

        # uncertainty_bias: (B, Dz, H, W)
        uncertainty_bias = self.uncertainty_proj(entropy)

        # height-aware gate:
        # 同时考虑高度层显式权重和不确定性偏置。
        gate_logits = self.height_refine(height_logits) + uncertainty_bias
        height_gate = torch.sigmoid(gate_logits)

        # 置信度调制:
        # 深度越确定，越相信 height_gate；
        # 深度越不确定，gate 越趋向保守的 0.5。
        height_gate = height_gate * confidence + 0.5 * (1.0 - confidence)

        # 将通道按高度层分组:
        # x_group: (B, Dz, group_channels, H, W)
        x_group = x.contiguous().view(
            B, self.Dz, self.group_channels, H, W
        )

        # 对每个高度组做重标定。
        recalibrated = x_group * height_gate.unsqueeze(2)

        # 恢复为 (B, C, H, W)
        recalibrated = recalibrated.contiguous().view(B, C, H, W)

        if self.use_residual:
            out = x + self.gamma * recalibrated
        else:
            out = recalibrated

        return out


@HEADS.register_module()
class BEVOCCHead2D_DUHR(BaseModule):
    """
    BEVOCCHead2D + DUHR.

    与原始 BEVOCCHead2D 的接口保持一致：
        in_dim
        out_dim
        Dz
        use_mask
        num_classes
        use_predicter
        class_balance
        loss_occ

    因此配置文件中只需要把:
        type='BEVOCCHead2D'
    改成:
        type='BEVOCCHead2D_DUHR'
    """

    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ=None,
                 duhr_reduction=4,
                 duhr_use_residual=True,
                 duhr_zero_init=True):
        super(BEVOCCHead2D_DUHR, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz

        # 新增：Depth-Uncertainty Guided Height-Aware Recalibration
        self.duhr = DUHRModule(
            channels=in_dim,
            Dz=Dz,
            reduction=duhr_reduction,
            use_residual=duhr_use_residual,
            zero_init=duhr_zero_init
        )

        out_channels = out_dim if use_predicter else num_classes * Dz

        # 保持原始 BEVOCCHead2D 的 final_conv 结构不变。
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )

        self.use_predicter = use_predicter

        # 保持原始 BEVOCCHead2D 的 predicter 结构不变。
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes

        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(
                1 / np.log(nusc_class_frequencies[:num_classes] + 0.001)
            )
            self.cls_weights = class_weights
            loss_occ['class_weight'] = class_weights

        self.loss_occ = build_loss(loss_occ)

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx)

        Returns:
            occ_pred: (B, Dx, Dy, Dz, num_classes)
        """

        # 新增：在 channel-to-height 解码之前做 DUHR 重标定。
        img_feats = self.duhr(img_feats)

        # 原始 BEVOCCHead2D 逻辑保持不变:
        # (B, C, Dy, Dx) -> (B, C, Dy, Dx) -> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]

        if self.use_predicter:
            # (B, Dx, Dy, C) -> (B, Dx, Dy, Dz*num_classes)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)
        else:
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, num_classes)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()

        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)

            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)

            if self.class_balance:
                valid_voxels = voxel_semantics[mask_camera.bool()]
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (
                        (valid_voxels == i).sum() * self.cls_weights[i]
                    )
            else:
                num_total_samples = mask_camera.sum()

            loss_occ = self.loss_occ(
                preds,
                voxel_semantics,
                mask_camera,
                avg_factor=num_total_samples
            )

            loss['loss_occ'] = loss_occ

        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)

            if self.class_balance:
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (
                        (voxel_semantics == i).sum() * self.cls_weights[i]
                    )
            else:
                num_total_samples = len(voxel_semantics)

            loss_occ = self.loss_occ(
                preds,
                voxel_semantics,
                avg_factor=num_total_samples
            )

            loss['loss_occ'] = loss_occ

        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)

        Returns:
            List[(Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)
        return list(occ_res)



class DUHRModuleV2(nn.Module):
    """
    DUHR-V2: Depth/Height-Uncertainty Guided Bidirectional Recalibration.

    相比 DUHR-V1:
    1. 使用双向重标定 scale = 1 + gamma * (2 * gate - 1)
    2. gate > 0.5 增强特征，gate < 0.5 抑制特征
    3. gamma 初始为 0.1，不再初始为 0
    4. 增加 channel gate，避免只做高度组粗粒度调制
    """

    def __init__(
        self,
        channels=256,
        Dz=16,
        reduction=4,
        gamma_init=0.1,
    ):
        super(DUHRModuleV2, self).__init__()

        assert channels % Dz == 0, (
            f"DUHRModuleV2 requires channels % Dz == 0, "
            f"but got channels={channels}, Dz={Dz}"
        )

        self.channels = channels
        self.Dz = Dz
        self.group_channels = channels // Dz

        mid_channels = max(channels // reduction, 32)
        se_channels = max(channels // 16, 16)

        self.context_conv = nn.Sequential(
            ConvModule(
                channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=None,
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=None,
                act_cfg=dict(type='ReLU')
            )
        )

        # 高度层预测分支: (B, Dz, H, W)
        self.height_logits = nn.Conv2d(mid_channels, Dz, kernel_size=1)

        # 不确定性分支：由高度分布 entropy 得到 uncertainty bias
        self.uncertainty_proj = nn.Sequential(
            nn.Conv2d(1, Dz, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(Dz, Dz, kernel_size=3, stride=1, padding=1, groups=Dz, bias=True)
        )

        # 通道注意力分支
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, se_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # 调制强度，初始 0.1，避免模块完全不起作用
        self.gamma = nn.Parameter(torch.ones(1) * gamma_init)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape

        context = self.context_conv(x)

        height_logits = self.height_logits(context)  # (B, Dz, H, W)
        height_prob = torch.softmax(height_logits, dim=1)

        entropy = -(height_prob * torch.log(height_prob.clamp_min(1e-6))).sum(
            dim=1, keepdim=True
        ) / math.log(self.Dz)

        uncertainty_bias = self.uncertainty_proj(entropy)
        height_gate = torch.sigmoid(height_logits + uncertainty_bias)

        height_gate = height_gate.unsqueeze(2).expand(
            B, self.Dz, self.group_channels, H, W
        ).contiguous().view(B, C, H, W)

        channel_gate = self.channel_gate(x)

        gate = height_gate * channel_gate

        # 双向重标定
        scale = 1.0 + self.gamma * (2.0 * gate - 1.0)

        out = x * scale
        return out


@HEADS.register_module()
class BEVOCCHead2D_DUHRV2(BEVOCCHead2D):
    """
    BEVOCCHead2D + DUHR-V2.

    继承原始 BEVOCCHead2D，只在 forward 之前增加 DUHR-V2。
    final_conv、predicter、loss、get_occ 全部复用原始 BEVOCCHead2D。
    """

    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ=None,
                 duhr_reduction=4,
                 duhr_gamma_init=0.1,
                 duhr_use_residual=True,
                 duhr_zero_init=False):
        super(BEVOCCHead2D_DUHRV2, self).__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            Dz=Dz,
            use_mask=use_mask,
            num_classes=num_classes,
            use_predicter=use_predicter,
            class_balance=class_balance,
            loss_occ=loss_occ
        )

        # 兼容旧配置参数
        if duhr_zero_init:
            duhr_gamma_init = 0.0

        self.duhr_v2 = DUHRModuleV2(
            channels=in_dim,
            Dz=Dz,
            reduction=duhr_reduction,
            gamma_init=duhr_gamma_init
        )

    def forward(self, img_feats):
        img_feats = self.duhr_v2(img_feats)
        return super(BEVOCCHead2D_DUHRV2, self).forward(img_feats)
