import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_neck
from .bevdet_occ import BEVDetOCC


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


@DETECTORS.register_module()
class BEVDetOCCMSBEV(BEVDetOCC):
    """
    FlashOcc + Multi-Scale LSS BEV Fusion.

    This detector keeps the original FlashOcc occupancy head unchanged.
    It extracts C2/C3/C4 image features, projects P8/P16/P32 features
    into the same BEV space through scale-specific LSS view transformers,
    and injects the multi-scale BEV feature into the original BEV feature
    through a zero-initialized residual adapter.

    Main path:
        P16 -> original LSS -> original BEV encoder -> F_base

    Multi-scale branch:
        P8  -> LSS(downsample=8)  -> BEV_8
        P16 -> original LSS       -> BEV_16
        P32 -> LSS(downsample=32) -> BEV_32

    Fusion:
        F_out = F_base + ResidualAdapter([BEV_8, BEV_16, BEV_32])

    The original BEVOCCHead2D is not replaced.
    """

    def __init__(self,
                 ms_view_transformer_p8=None,
                 ms_view_transformer_p32=None,
                 ms_img_channels=(512, 1024, 2048),
                 ms_proj_channels=256,
                 ms_bev_channels=64,
                 ms_fusion_channels=256,
                 ms_use_p8=True,
                 ms_use_p16=True,
                 ms_use_p32=True,
                 **kwargs):
        super(BEVDetOCCMSBEV, self).__init__(**kwargs)

        self.ms_use_p8 = ms_use_p8
        self.ms_use_p16 = ms_use_p16
        self.ms_use_p32 = ms_use_p32
        self.ms_bev_channels = ms_bev_channels

        if self.ms_use_p8:
            assert ms_view_transformer_p8 is not None
            self.ms_p8_proj = ConvBNReLU(ms_img_channels[0], ms_proj_channels, kernel_size=1)
            self.ms_view_transformer_p8 = build_neck(ms_view_transformer_p8)

        if self.ms_use_p32:
            assert ms_view_transformer_p32 is not None
            self.ms_p32_proj = ConvBNReLU(ms_img_channels[2], ms_proj_channels, kernel_size=1)
            self.ms_view_transformer_p32 = build_neck(ms_view_transformer_p32)

        num_ms = int(ms_use_p8) + int(ms_use_p16) + int(ms_use_p32)
        assert num_ms > 0

        self.ms_fusion = nn.Sequential(
            nn.Conv2d(num_ms * ms_bev_channels, ms_fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(ms_fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(ms_fusion_channels, ms_fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ms_fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(ms_fusion_channels, ms_fusion_channels, kernel_size=3, padding=1, bias=True)
        )

        # Zero-init residual adapter:
        # at the beginning, F_out = F_base, so the model starts from the baseline.
        nn.init.constant_(self.ms_fusion[-1].weight, 0.0)
        nn.init.constant_(self.ms_fusion[-1].bias, 0.0)

    def _format_img_feat(self, x, B, N):
        """
        Convert image feature from (B*N, C, H, W) to (B, N, C, H, W).
        """
        _, C, H, W = x.shape
        return x.view(B, N, C, H, W)

    def _view_transform(self, transformer, feat, inputs):
        """
        Run LSS view transformer and support both return styles.
        """
        out = transformer([feat] + inputs[1:7])
        if isinstance(out, tuple):
            return out[0], out[1]
        return out, None

    def _bev_encoder(self, x):
        """
        Same as BEVDet BEV encoder path.
        """
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    def extract_img_feat_msbev(self, img_inputs):
        """
        Extract original FlashOcc BEV feature and multi-scale BEV residual.

        Args:
            img_inputs: input list from PrepareImageInputs.

        Returns:
            img_feats: [F_out], where F_out is fused BEV feature.
            depth: original P16 depth distribution from the main LSS branch.
        """
        inputs = self.prepare_inputs(img_inputs)
        imgs = inputs[0]  # (B, N, C, H, W)

        B, N, C, H, W = imgs.shape
        imgs_flat = imgs.view(B * N, C, H, W)

        # Backbone must output C2/C3/C4 when config uses out_indices=(1,2,3).
        feats = self.img_backbone(imgs_flat)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]

        assert len(feats) >= 3, \
            'MS-BEV requires img_backbone out_indices=(1,2,3), producing C2/C3/C4.'

        c2, c3, c4 = feats[-3], feats[-2], feats[-1]

        # Original FlashOcc main branch:
        # use C3/C4 through original img_neck, keeping original M1 path.
        p16 = self.img_neck([c3, c4])
        if isinstance(p16, (list, tuple)):
            p16 = p16[0]

        p16_5d = self._format_img_feat(p16, B, N)
        bev_16, depth = self._view_transform(self.img_view_transformer, p16_5d, inputs)

        # Original BEV encoder output, i.e., F_base.
        f_base = self._bev_encoder(bev_16)

        ms_bev_list = []

        # P8 branch: C2 -> 256 channels -> LSS downsample=8.
        if self.ms_use_p8:
            p8 = self.ms_p8_proj(c2)
            p8_5d = self._format_img_feat(p8, B, N)
            bev_8, _ = self._view_transform(self.ms_view_transformer_p8, p8_5d, inputs)
            ms_bev_list.append(bev_8)

        # P16 branch: reuse original camera BEV before BEV encoder.
        if self.ms_use_p16:
            ms_bev_list.append(bev_16)

        # P32 branch: C4 -> 256 channels -> LSS downsample=32.
        if self.ms_use_p32:
            p32 = self.ms_p32_proj(c4)
            p32_5d = self._format_img_feat(p32, B, N)
            bev_32, _ = self._view_transform(self.ms_view_transformer_p32, p32_5d, inputs)
            ms_bev_list.append(bev_32)

        # Spatial alignment safeguard.
        target_size = f_base.shape[-2:]
        aligned_ms_bev = []
        for feat in ms_bev_list:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            aligned_ms_bev.append(feat)

        ms_bev = torch.cat(aligned_ms_bev, dim=1)
        delta = self.ms_fusion(ms_bev)

        f_out = f_base + delta

        return [f_out], depth

    def extract_feat(self,
                     points=None,
                     img_inputs=None,
                     img_metas=None,
                     **kwargs):
        img_feats, depth = self.extract_img_feat_msbev(img_inputs)
        pts_feats = None
        return img_feats, pts_feats, depth

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
        img_feats, pts_feats, depth = self.extract_feat(
            points=points,
            img_inputs=img_inputs,
            img_metas=img_metas,
            **kwargs
        )

        losses = dict()

        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(
                occ_bev_feature,
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            )

        loss_occ = self.forward_occ_train(
            occ_bev_feature,
            voxel_semantics,
            mask_camera
        )
        losses.update(loss_occ)
        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        img_feats, _, _ = self.extract_feat(
            points=points,
            img_inputs=img,
            img_metas=img_metas,
            **kwargs
        )

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(
                occ_bev_feature,
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            )

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)
        return occ_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points=points,
            img_inputs=img_inputs,
            img_metas=img_metas,
            **kwargs
        )
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(
                occ_bev_feature,
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            )
        outs = self.occ_head(occ_bev_feature)
        return outs
