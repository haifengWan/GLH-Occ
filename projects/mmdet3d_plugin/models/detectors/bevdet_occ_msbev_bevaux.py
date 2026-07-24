import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models import DETECTORS
from .bevdet_occ_msbev import BEVDetOCCMSBEV


@DETECTORS.register_module()
class BEVDetOCCMSBEVBEVAux(BEVDetOCCMSBEV):
    """
    FlashOcc + MS-BEV + BEVAux.

    This detector combines:
        1. Multi-Scale LSS BEV Fusion:
            multi-scale image features -> multi-scale LSS projection -> fused BEV feature

        2. BEV-level auxiliary semantic supervision:
            fused BEV feature -> BEVAux head -> loss_bev_aux

    Main occupancy path:
        F_out -> original BEVOCCHead2D -> loss_occ

    Auxiliary path:
        F_out -> BEVAux head -> loss_bev_aux

    Total loss:
        loss = loss_occ + loss_bev_aux

    Note:
        - The original BEVOCCHead2D is not replaced.
        - DepthAux is not used.
        - BEVAux is training-only and is not used in simple_test.
        - Inference path is inherited from BEVDetOCCMSBEV.
    """

    def __init__(self,
                 bev_aux_loss_weight=0.05,
                 bev_aux_in_channels=256,
                 bev_aux_hidden_channels=256,
                 bev_aux_num_classes=17,
                 bev_aux_ignore_idx=255,
                 bev_aux_pos_weight_max=3.0,
                 **kwargs):
        super(BEVDetOCCMSBEVBEVAux, self).__init__(**kwargs)

        self.bev_aux_loss_weight = bev_aux_loss_weight
        self.bev_aux_num_classes = bev_aux_num_classes
        self.bev_aux_ignore_idx = bev_aux_ignore_idx
        self.bev_aux_pos_weight_max = bev_aux_pos_weight_max

        self.bev_aux_head = nn.Sequential(
            nn.Conv2d(
                bev_aux_in_channels,
                bev_aux_hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(bev_aux_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                bev_aux_hidden_channels,
                bev_aux_num_classes,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

    def build_bev_multihot_target(self, voxel_semantics, mask_camera):
        """
        Build BEV-level multi-hot semantic target from voxel-level occupancy GT.

        Args:
            voxel_semantics: Tensor, shape (B, Dx, Dy, Dz)
                Semantic occupancy labels.
                Usually:
                    0-16: semantic classes
                    17: free / empty
                    255: ignore

            mask_camera: Tensor, shape (B, Dx, Dy, Dz)
                Camera-visible mask.

        Returns:
            target: Tensor, shape (B, 17, Dx, Dy)
                Multi-hot BEV semantic target, excluding free class.

            bev_mask: Tensor, shape (B, Dx, Dy)
                Valid BEV supervision mask.
        """
        sem = voxel_semantics.long()
        valid = (mask_camera > 0) & (sem != self.bev_aux_ignore_idx)

        # A BEV cell is valid if at least one visible voxel exists in its height column.
        bev_mask = valid.any(dim=-1).float()  # (B, Dx, Dy)

        # Classes 0-16 are semantic classes. Class 17/free is not a positive class.
        target_list = []
        for cls_idx in range(self.bev_aux_num_classes):
            cls_target = ((sem == cls_idx) & valid).any(dim=-1).float()
            target_list.append(cls_target)

        target = torch.stack(target_list, dim=1)  # (B, 17, Dx, Dy)
        return target, bev_mask

    def loss_bev_aux(self, bev_logits, voxel_semantics, mask_camera):
        """
        Compute BEV auxiliary multi-label semantic loss.

        Args:
            bev_logits: Tensor, shape (B, 17, H, W)
            voxel_semantics: Tensor, shape (B, Dx, Dy, Dz)
            mask_camera: Tensor, shape (B, Dx, Dy, Dz)

        Returns:
            Weighted scalar loss.
        """
        target, bev_mask = self.build_bev_multihot_target(
            voxel_semantics,
            mask_camera
        )

        # Align target to feature spatial size if necessary.
        if bev_logits.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(
                target,
                size=bev_logits.shape[-2:],
                mode='nearest'
            )
            bev_mask = F.interpolate(
                bev_mask.unsqueeze(1),
                size=bev_logits.shape[-2:],
                mode='nearest'
            ).squeeze(1)

        mask = bev_mask.unsqueeze(1)  # (B, 1, H, W)

        bce = F.binary_cross_entropy_with_logits(
            bev_logits,
            target,
            reduction='none'
        )

        # Dynamic positive reweighting for sparse BEV semantic positives.
        with torch.no_grad():
            valid_count = mask.sum(dim=(0, 2, 3)).clamp_min(1.0)  # (1,)
            pos_count = (target * mask).sum(dim=(0, 2, 3))        # (C,)
            neg_count = valid_count - pos_count

            pos_weight = neg_count / pos_count.clamp_min(1.0)
            pos_weight = pos_weight.clamp(
                min=1.0,
                max=self.bev_aux_pos_weight_max
            ).view(1, -1, 1, 1)

        weight = torch.where(
            target > 0.5,
            pos_weight,
            torch.ones_like(target)
        )

        bce = bce * weight

        denom = (mask.sum() * self.bev_aux_num_classes).clamp_min(1.0)
        loss = (bce * mask).sum() / denom

        return self.bev_aux_loss_weight * loss

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """
        Training forward.

        The feature used by both loss_occ and loss_bev_aux is the MS-BEV enhanced
        BEV feature returned by self.extract_feat(), inherited from BEVDetOCCMSBEV.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points=points,
            img_inputs=img_inputs,
            img_metas=img_metas,
            **kwargs
        )

        losses = dict()

        voxel_semantics = kwargs['voxel_semantics']  # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']          # (B, Dx, Dy, Dz)

        occ_bev_feature = img_feats[0]

        # Keep the same occupancy input as original FlashOcc/MS-BEV.
        if self.upsample:
            occ_bev_feature = F.interpolate(
                occ_bev_feature,
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            )

        # 1. Original FlashOcc occupancy loss.
        loss_occ = self.forward_occ_train(
            occ_bev_feature,
            voxel_semantics,
            mask_camera
        )
        losses.update(loss_occ)

        # 2. Training-only BEV auxiliary semantic loss on the same fused BEV feature.
        if self.bev_aux_loss_weight > 0:
            bev_logits = self.bev_aux_head(occ_bev_feature)
            losses['loss_bev_aux'] = self.loss_bev_aux(
                bev_logits,
                voxel_semantics,
                mask_camera
            )

        return losses
