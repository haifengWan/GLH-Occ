import torch

from mmdet3d.models import DETECTORS

from .bevdet_occ import BEVDetOCC
from .bevdet_occ_msbev_lhcr_bevaux import LatentHeightColumnReasoner


@DETECTORS.register_module()
class BEVDetOCCLHCR(BEVDetOCC):
    """FlashOcc M1 with LHCR only.

    The original FlashOcc image-to-BEV path and BEVOCCHead2D are retained.
    LHCR is inserted after the original BEV encoder and before the
    occupancy head. MS-BEV and BEV-Aux are not constructed.
    """

    def __init__(
        self,
        lhcr_in_channels=256,
        lhcr_latent_height=16,
        lhcr_height_channels=32,
        lhcr_num_height_blocks=1,
        lhcr_num_local_blocks=1,
        lhcr_num_global_blocks=1,
        lhcr_global_num_prototypes=16,
        lhcr_global_pool_size=8,
        lhcr_mlp_ratio=2.0,
        lhcr_dropout=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.lhcr_encoder = LatentHeightColumnReasoner(
            in_channels=lhcr_in_channels,
            latent_height=lhcr_latent_height,
            height_channels=lhcr_height_channels,
            num_height_blocks=lhcr_num_height_blocks,
            num_local_blocks=lhcr_num_local_blocks,
            num_global_blocks=lhcr_num_global_blocks,
            global_num_prototypes=lhcr_global_num_prototypes,
            global_pool_size=lhcr_global_pool_size,
            mlp_ratio=lhcr_mlp_ratio,
            dropout=lhcr_dropout,
        )

    def extract_feat(self, *args, **kwargs):
        img_feats, pts_feats, depth = super().extract_feat(*args, **kwargs)

        img_feats_is_tuple = isinstance(img_feats, tuple)
        img_feats = list(img_feats)

        if len(img_feats) == 0:
            raise RuntimeError("BEVDetOCCLHCR received an empty img_feats list.")

        bev_feat = img_feats[0]
        if not torch.is_tensor(bev_feat) or bev_feat.ndim != 4:
            raise TypeError(
                "LHCR expects img_feats[0] to be a 4D BEV tensor "
                "(B, C, H, W), but got {}".format(type(bev_feat))
            )

        img_feats[0] = self.lhcr_encoder(bev_feat)

        if img_feats_is_tuple:
            img_feats = tuple(img_feats)

        return img_feats, pts_feats, depth
