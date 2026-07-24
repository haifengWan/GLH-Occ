import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_neck

from .bevdet_occ import BEVStereo4DOCC


class ConvBNReLU(nn.Sequential):
    """Conv-BN-ReLU projection used by the auxiliary image scales."""

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


@DETECTORS.register_module()
class BEVStereo4DOCCMSBEVBEVAux(BEVStereo4DOCC):
    """
    FlashOcc M3 + multi-scale BEV fusion + BEV auxiliary supervision.

    The original M3 path is preserved:
        Swin-B -> BEV-Stereo -> temporal BEV fusion -> BEV encoder -> F_base

    The additional key-frame path is:
        P8/P32 -> lightweight LSS branches
        key-frame P16 BEV -> reused from the original M3 path
        [BEV_P8, BEV_P16, BEV_P32] -> zero-initialized residual adapter

    The final BEV feature is:
        F_out = F_base + Delta_F_ms

    During training only:
        F_out -> BEV auxiliary head -> loss_bev_aux

    Notes:
        1. P8/P32 are extracted only from the key frame.
        2. The original stereo depth loss and occupancy loss are retained.
        3. The occupancy head and inference output format are unchanged.
    """

    def __init__(
        self,
        ms_view_transformer_p8=None,
        ms_view_transformer_p32=None,
        ms_img_channels=(256, 512, 1024),
        ms_proj_channels=256,
        ms_bev_channels=80,
        ms_fusion_channels=256,
        ms_use_p8=True,
        ms_use_p16=True,
        ms_use_p32=True,
        bev_aux_loss_weight=0.05,
        bev_aux_in_channels=256,
        bev_aux_hidden_channels=256,
        bev_aux_num_classes=17,
        bev_aux_ignore_idx=255,
        bev_aux_pos_weight_max=3.0,
        ms_debug=False,
        **kwargs
    ):
        super(BEVStereo4DOCCMSBEVBEVAux, self).__init__(**kwargs)

        self.ms_use_p8 = ms_use_p8
        self.ms_use_p16 = ms_use_p16
        self.ms_use_p32 = ms_use_p32
        self.ms_bev_channels = ms_bev_channels
        self.ms_debug = ms_debug
        self._ms_debug_printed = False

        # Swin-B stage channels after removing the P4 stereo feature:
        # P8=256, P16=512, P32=1024.
        self.ms_img_channels = tuple(ms_img_channels)

        if self.ms_use_p8:
            assert ms_view_transformer_p8 is not None
            self.ms_p8_proj = ConvBNReLU(
                self.ms_img_channels[0],
                ms_proj_channels,
                kernel_size=1,
            )
            self.ms_view_transformer_p8 = build_neck(
                ms_view_transformer_p8
            )

        if self.ms_use_p32:
            assert ms_view_transformer_p32 is not None
            self.ms_p32_proj = ConvBNReLU(
                self.ms_img_channels[2],
                ms_proj_channels,
                kernel_size=1,
            )
            self.ms_view_transformer_p32 = build_neck(
                ms_view_transformer_p32
            )

        num_ms_scales = (
            int(self.ms_use_p8)
            + int(self.ms_use_p16)
            + int(self.ms_use_p32)
        )
        assert num_ms_scales > 0

        self.ms_fusion = nn.Sequential(
            nn.Conv2d(
                num_ms_scales * ms_bev_channels,
                ms_fusion_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(ms_fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                ms_fusion_channels,
                ms_fusion_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ms_fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                ms_fusion_channels,
                ms_fusion_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

        # Keep the initial model identical to the M3 baseline.
        nn.init.constant_(self.ms_fusion[-1].weight, 0.0)
        nn.init.constant_(self.ms_fusion[-1].bias, 0.0)

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
                bias=False,
            ),
            nn.BatchNorm2d(bev_aux_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                bev_aux_hidden_channels,
                bev_aux_num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        # Temporary caches. In the official BEVStereo4D loop, frames are
        # processed from the oldest/reference side to fid=0, so the final
        # non-reference call is the key frame.
        self._cached_key_p8 = None
        self._cached_key_p32 = None
        self._cached_key_bev_p16 = None

    def _reset_ms_cache(self):
        self._cached_key_p8 = None
        self._cached_key_p32 = None
        self._cached_key_bev_p16 = None

    def image_encoder(self, img, stereo=False):
        """
        Preserve the original M3 image encoder while retaining key-frame
        P8/P32 tensors for the auxiliary LSS branches.

        With the supplied configuration, Swin-B returns:
            [P4_stereo, P8, P16, P32]
        when stereo=True.
        """
        imgs = img
        batch_size, num_cams, channels, height, width = imgs.shape
        imgs = imgs.view(
            batch_size * num_cams,
            channels,
            height,
            width,
        )

        feats = self.img_backbone(imgs)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        else:
            feats = list(feats)

        stereo_feat = None
        if stereo:
            if len(feats) < 4:
                raise RuntimeError(
                    "M3 MS-BEV expects Swin-B outputs "
                    "[P4_stereo, P8, P16, P32]. Check "
                    "return_stereo_feat=True and out_indices=(1, 2, 3)."
                )
            stereo_feat = feats[0]
            pyramid_feats = feats[1:]
        else:
            pyramid_feats = feats

        if len(pyramid_feats) < 3:
            raise RuntimeError(
                "MS-BEV requires three pyramid features P8/P16/P32."
            )

        # Every normal prepare_bev_feat call overwrites these tensors.
        # Because the official frame loop processes fid=0 last, the cache
        # after super().extract_img_feat() corresponds to the key frame.
        self._cached_key_p8 = pyramid_feats[-3]
        self._cached_key_p32 = pyramid_feats[-1]

        x = pyramid_feats
        if self.with_img_neck:
            x = self.img_neck(x)
            if isinstance(x, (list, tuple)):
                x = x[0]

        _, out_channels, out_h, out_w = x.shape
        x = x.view(
            batch_size,
            num_cams,
            out_channels,
            out_h,
            out_w,
        )
        return x, stereo_feat

    def prepare_bev_feat(
        self,
        img,
        sensor2keyego,
        ego2global,
        intrin,
        post_rot,
        post_tran,
        bda,
        mlp_input,
        feat_prev_iv,
        k2s_sensor,
        extra_ref_frame,
    ):
        """
        Run the original BEV-Stereo transformation and cache the last
        non-reference BEV tensor, which is the key-frame P16 BEV tensor.
        """
        bev_feat, depth, stereo_feat = super(
            BEVStereo4DOCCMSBEVBEVAux,
            self,
        ).prepare_bev_feat(
            img,
            sensor2keyego,
            ego2global,
            intrin,
            post_rot,
            post_tran,
            bda,
            mlp_input,
            feat_prev_iv,
            k2s_sensor,
            extra_ref_frame,
        )

        if (not extra_ref_frame) and (bev_feat is not None):
            self._cached_key_bev_p16 = bev_feat

        return bev_feat, depth, stereo_feat

    @staticmethod
    def _format_img_feat(x, batch_size, num_cams):
        """Convert (B*N,C,H,W) to (B,N,C,H,W)."""
        _, channels, height, width = x.shape
        return x.view(
            batch_size,
            num_cams,
            channels,
            height,
            width,
        )

    @staticmethod
    def _view_transform(transformer, feat, geometry_inputs):
        """
        Apply an ordinary LSS view transformer.

        geometry_inputs:
            [sensor2keyego, ego2global, intrin,
             post_rot, post_tran, bda]
        """
        out = transformer([feat] + list(geometry_inputs))
        if isinstance(out, tuple):
            return out[0], out[1]
        return out, None

    def _get_keyframe_geometry(self, img_inputs, sequential=False):
        """
        Return key-frame camera geometry in the format required by
        LSSViewTransformer.
        """
        if sequential:
            # Format constructed by BEVStereo4D.extract_img_feat(...,
            # sequential=True).
            sensor2keyego = img_inputs[1][0:1, ...]
            ego2global = img_inputs[2][0:1, ...]
            intrin = img_inputs[3]
            post_rot = img_inputs[6]
            post_tran = img_inputs[7]
            bda = img_inputs[8][0:1, ...]
        else:
            prepared = self.prepare_inputs(img_inputs, stereo=True)
            _, sensor2keyegos, ego2globals, intrins, post_rots, \
                post_trans, bda, _ = prepared

            sensor2keyego = sensor2keyegos[0]
            ego2global = ego2globals[0]
            intrin = intrins[0]
            post_rot = post_rots[0]
            post_tran = post_trans[0]

        return [
            sensor2keyego,
            ego2global,
            intrin,
            post_rot,
            post_tran,
            bda,
        ]

    def _fuse_ms_bev(self, f_base, img_inputs, sequential=False):
        """Build P8/P16/P32 BEV features and add the residual adapter."""
        if self._cached_key_p8 is None or self._cached_key_p32 is None:
            raise RuntimeError(
                "P8/P32 cache is empty. Check the Swin-B output indices "
                "and the overridden image_encoder()."
            )

        if self.ms_use_p16 and self._cached_key_bev_p16 is None:
            raise RuntimeError(
                "Key-frame P16 BEV cache is empty. Check "
                "prepare_bev_feat() and the M3 temporal path."
            )

        geometry_inputs = self._get_keyframe_geometry(
            img_inputs,
            sequential=sequential,
        )
        sensor2keyego = geometry_inputs[0]
        batch_size, num_cams = sensor2keyego.shape[:2]

        ms_bev_list = []

        if self.ms_use_p8:
            p8 = self.ms_p8_proj(self._cached_key_p8)
            p8 = self._format_img_feat(p8, batch_size, num_cams)
            bev_p8, _ = self._view_transform(
                self.ms_view_transformer_p8,
                p8,
                geometry_inputs,
            )
            ms_bev_list.append(bev_p8)

        if self.ms_use_p16:
            ms_bev_list.append(self._cached_key_bev_p16)

        if self.ms_use_p32:
            p32 = self.ms_p32_proj(self._cached_key_p32)
            p32 = self._format_img_feat(p32, batch_size, num_cams)
            bev_p32, _ = self._view_transform(
                self.ms_view_transformer_p32,
                p32,
                geometry_inputs,
            )
            ms_bev_list.append(bev_p32)

        target_size = f_base.shape[-2:]
        aligned = []
        for feat in ms_bev_list:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode="bilinear",
                    align_corners=True,
                )
            aligned.append(feat)

        ms_bev = torch.cat(aligned, dim=1)
        delta = self.ms_fusion(ms_bev)

        if delta.shape != f_base.shape:
            raise RuntimeError(
                "MS-BEV residual shape mismatch: "
                f"delta={tuple(delta.shape)}, "
                f"f_base={tuple(f_base.shape)}. "
                "Check ms_bev_channels and ms_fusion_channels."
            )

        if self.ms_debug and not self._ms_debug_printed:
            print(
                "[M3-MSBEV] "
                f"P8={tuple(self._cached_key_p8.shape)}, "
                f"P32={tuple(self._cached_key_p32.shape)}, "
                f"P16_BEV={tuple(self._cached_key_bev_p16.shape)}, "
                f"MS_BEV={tuple(ms_bev.shape)}, "
                f"F_base={tuple(f_base.shape)}, "
                f"delta={tuple(delta.shape)}"
            )
            self._ms_debug_printed = True

        return f_base + delta

    def extract_img_feat(
        self,
        img_inputs,
        img_metas,
        pred_prev=False,
        sequential=False,
        **kwargs
    ):
        """
        Preserve the complete M3 feature extraction, then inject MS-BEV
        after the temporal BEV encoder.
        """
        self._reset_ms_cache()

        result = super(
            BEVStereo4DOCCMSBEVBEVAux,
            self,
        ).extract_img_feat(
            img_inputs,
            img_metas,
            pred_prev=pred_prev,
            sequential=sequential,
            **kwargs
        )

        # pred_prev is the special first stage of sequential inference and
        # returns history features plus prepared inputs, not final BEV features.
        if pred_prev:
            return result

        img_feats, depth = result
        f_base = img_feats[0]
        f_out = self._fuse_ms_bev(
            f_base,
            img_inputs,
            sequential=sequential,
        )
        return [f_out], depth

    def build_bev_multihot_target(self, voxel_semantics, mask_camera):
        """Build the same 17-class BEV multi-hot target used by M1."""
        sem = voxel_semantics.long()
        valid = (
            (mask_camera > 0)
            & (sem != self.bev_aux_ignore_idx)
        )

        bev_mask = valid.any(dim=-1).float()

        target_list = []
        for cls_idx in range(self.bev_aux_num_classes):
            cls_target = (
                (sem == cls_idx)
                & valid
            ).any(dim=-1).float()
            target_list.append(cls_target)

        target = torch.stack(target_list, dim=1)
        return target, bev_mask

    def loss_bev_aux(
        self,
        bev_logits,
        voxel_semantics,
        mask_camera,
    ):
        """Compute the same dynamically reweighted BCE loss used by M1."""
        target, bev_mask = self.build_bev_multihot_target(
            voxel_semantics,
            mask_camera,
        )

        if bev_logits.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(
                target,
                size=bev_logits.shape[-2:],
                mode="nearest",
            )
            bev_mask = F.interpolate(
                bev_mask.unsqueeze(1),
                size=bev_logits.shape[-2:],
                mode="nearest",
            ).squeeze(1)

        mask = bev_mask.unsqueeze(1)

        bce = F.binary_cross_entropy_with_logits(
            bev_logits,
            target,
            reduction="none",
        )

        with torch.no_grad():
            valid_count = mask.sum(
                dim=(0, 2, 3)
            ).clamp_min(1.0)
            pos_count = (
                target * mask
            ).sum(dim=(0, 2, 3))
            neg_count = valid_count - pos_count

            pos_weight = (
                neg_count / pos_count.clamp_min(1.0)
            )
            pos_weight = pos_weight.clamp(
                min=1.0,
                max=self.bev_aux_pos_weight_max,
            ).view(1, -1, 1, 1)

        weight = torch.where(
            target > 0.5,
            pos_weight,
            torch.ones_like(target),
        )
        bce = bce * weight

        denom = (
            mask.sum() * self.bev_aux_num_classes
        ).clamp_min(1.0)
        loss = (bce * mask).sum() / denom

        return self.bev_aux_loss_weight * loss

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img_inputs=None,
        proposals=None,
        gt_bboxes_ignore=None,
        **kwargs
    ):
        """
        M3 training objective:
            loss_depth + loss_occ + loss_bev_aux
        """
        img_feats, _, depth = self.extract_feat(
            points,
            img_inputs=img_inputs,
            img_metas=img_metas,
            **kwargs
        )

        losses = dict()

        gt_depth = kwargs["gt_depth"]
        losses["loss_depth"] = (
            self.img_view_transformer.get_depth_loss(
                gt_depth,
                depth,
            )
        )

        voxel_semantics = kwargs["voxel_semantics"]
        mask_camera = kwargs["mask_camera"]

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(
                occ_bev_feature,
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )

        losses.update(
            self.forward_occ_train(
                occ_bev_feature,
                voxel_semantics,
                mask_camera,
            )
        )

        if self.bev_aux_loss_weight > 0:
            bev_logits = self.bev_aux_head(
                occ_bev_feature
            )
            losses["loss_bev_aux"] = self.loss_bev_aux(
                bev_logits,
                voxel_semantics,
                mask_camera,
            )

        return losses

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Decode occupancy predictions without depending on get_occ_gpu().

        Direct argmax on logits is equivalent to softmax(...).argmax(...),
        while preserving the output format expected by the Occ3D evaluator.
        This override affects only this M3 extension class.
        """
        outs = self.occ_head(img_feats)
        occ_preds = outs.argmax(dim=-1)
        occ_preds = occ_preds.cpu().numpy().astype('uint8')
        return list(occ_preds)

