import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import DETECTORS

from .bevdet_occ_msbev_bevaux import BEVDetOCCMSBEVBEVAux


def _make_gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class HeightColumnMixer(nn.Module):
    """
    Height-column mixer.

    Input:
        x: (B, C_h, Z_l, H, W)

    Purpose:
        Model vertical dependency inside each latent BEV column.
    """

    def __init__(self, channels, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)

        self.norm = _make_gn(channels)
        self.dw_z = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False,
        )
        self.ffn = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
        )
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.norm(x)
        y = self.dw_z(y)
        y = self.ffn(y)
        return x + self.drop(y)


class LocalColumnSpatialMixer(nn.Module):
    """
    Local column-spatial mixer.

    Input:
        x: (B, C_h, Z_l, H, W)

    Purpose:
        Model local spatial dependency among neighboring BEV columns
        while preserving latent height structure.
    """

    def __init__(self, channels, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)

        self.norm = _make_gn(channels)
        self.dw_xy = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=channels,
            bias=False,
        )
        self.ffn = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
        )
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.norm(x)
        y = self.dw_xy(y)
        y = self.ffn(y)
        return x + self.drop(y)


class GlobalPrototypeContext(nn.Module):
    """
    Compact global prototype context.

    Instead of dense BEV-to-BEV attention, this module compresses
    low-resolution latent-height BEV tokens into K scene prototypes,
    then retrieves global context from those prototypes.

    Input:
        x: (B, C_h, Z_l, H, W)
    """

    def __init__(
        self,
        channels,
        num_prototypes=16,
        pool_size=8,
        dropout=0.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_prototypes = num_prototypes
        self.pool_size = pool_size
        self.scale = channels ** -0.5

        self.norm = _make_gn(channels)

        self.prototype_queries = nn.Parameter(
            torch.randn(num_prototypes, channels) * 0.02
        )

        self.key_low = nn.Linear(channels, channels, bias=False)
        self.value_low = nn.Linear(channels, channels, bias=False)

        self.query_token = nn.Linear(channels, channels, bias=False)
        self.key_proto = nn.Linear(channels, channels, bias=False)
        self.value_proto = nn.Linear(channels, channels, bias=False)

        self.out_proj = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            _make_gn(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=1, bias=True),
        )

        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        x: (B, C, Z, H, W)
        """
        b, c, z, h, w = x.shape
        identity = x

        x_norm = self.norm(x)

        ph = min(self.pool_size, h)
        pw = min(self.pool_size, w)

        # Low-resolution latent-height tokens.
        # Shape: (B, C, Z, ph, pw)
        x_low = F.adaptive_avg_pool3d(x_norm, output_size=(z, ph, pw))

        # Shape: (B, M, C), M = Z * ph * pw
        tokens = x_low.permute(0, 2, 3, 4, 1).contiguous().view(b, -1, c)

        k_low = self.key_low(tokens)
        v_low = self.value_low(tokens)

        # Learn global prototypes.
        # Shape: (B, K, C)
        q_proto = self.prototype_queries.unsqueeze(0).expand(b, -1, -1)

        # Shape: (B, K, M)
        assign = torch.softmax(
            torch.einsum('bkc,bmc->bkm', q_proto, k_low) * self.scale,
            dim=-1,
        )

        # Shape: (B, K, C)
        prototypes = torch.einsum('bkm,bmc->bkc', assign, v_low)

        # Retrieve global context for low-resolution tokens.
        q_token = self.query_token(tokens)
        k_proto = self.key_proto(prototypes)
        v_proto = self.value_proto(prototypes)

        # Shape: (B, M, K)
        attn = torch.softmax(
            torch.einsum('bmc,bkc->bmk', q_token, k_proto) * self.scale,
            dim=-1,
        )

        # Shape: (B, M, C)
        context = torch.einsum('bmk,bkc->bmc', attn, v_proto)

        # Back to low-resolution latent map.
        context = context.view(b, z, ph, pw, c).permute(0, 4, 1, 2, 3).contiguous()

        # Restore to original latent resolution.
        context = F.interpolate(
            context,
            size=(z, h, w),
            mode='trilinear',
            align_corners=False,
        )

        context = self.out_proj(context)
        return identity + self.drop(context)


class LatentHeightColumnReasoner(nn.Module):
    """
    LHCR-Full: Latent Height Column Reasoning.

    Input:
        bev_feat: (B, 256, H, W)

    Main idea:
        2D BEV channels are not treated as unordered feature dimensions.
        They are projected into latent height columns:
            (B, 256, H, W) -> (B, 32, 16, H, W)

        where:
            Z_l = 16 aligns with the 16 output occupancy height bins.
            C_h = 32 gives each latent height state sufficient capacity.

    Reasoning stages:
        1. Height-column mixing
        2. Local column-spatial mixing
        3. Compact global prototype context
        4. Zero-init gated residual back to BEV feature
    """

    def __init__(
        self,
        in_channels=256,
        latent_height=16,
        height_channels=32,
        num_height_blocks=1,
        num_local_blocks=1,
        num_global_blocks=1,
        global_num_prototypes=16,
        global_pool_size=8,
        mlp_ratio=2.0,
        dropout=0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.latent_height = latent_height
        self.height_channels = height_channels
        self.latent_dim = latent_height * height_channels

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, self.latent_dim, kernel_size=1, bias=False),
            _make_gn(self.latent_dim),
            nn.GELU(),
        )

        self.height_mixer = nn.Sequential(*[
            HeightColumnMixer(
                channels=height_channels,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_height_blocks)
        ])

        self.local_mixer = nn.Sequential(*[
            LocalColumnSpatialMixer(
                channels=height_channels,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_local_blocks)
        ])

        self.global_mixer = nn.Sequential(*[
            GlobalPrototypeContext(
                channels=height_channels,
                num_prototypes=global_num_prototypes,
                pool_size=global_pool_size,
                dropout=dropout,
            )
            for _ in range(num_global_blocks)
        ])

        self.latent_fuse = nn.Sequential(
            nn.Conv3d(height_channels * 4, height_channels, kernel_size=1, bias=False),
            _make_gn(height_channels),
            nn.GELU(),
            nn.Conv3d(height_channels, height_channels, kernel_size=1, bias=True),
        )

        self.delta_mid = nn.Sequential(
            nn.Conv2d(self.latent_dim, in_channels, kernel_size=1, bias=False),
            _make_gn(in_channels),
            nn.GELU(),
        )

        self.delta_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

        self.gate = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=True)

        self._init_zero_residual()

    def _init_zero_residual(self):
        # Initial output is identity because delta_out is zero.
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)

        # Initial gate is sigmoid(0)=0.5, but delta is zero.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, bev_feat):
        b, c, h, w = bev_feat.shape
        assert c == self.in_channels, \
            'LHCR expects {} channels, got {}'.format(self.in_channels, c)

        identity = bev_feat

        # (B, latent_dim, H, W)
        x = self.input_proj(bev_feat)

        # (B, C_h, Z_l, H, W)
        x = x.view(
            b,
            self.latent_height,
            self.height_channels,
            h,
            w,
        ).permute(0, 2, 1, 3, 4).contiguous()

        x_height = self.height_mixer(x)
        x_local = self.local_mixer(x_height)
        x_global = self.global_mixer(x_local)

        x_fuse = torch.cat([x, x_height, x_local, x_global], dim=1)
        x_fuse = self.latent_fuse(x_fuse)

        # Back to BEV channel representation.
        x_fuse = x_fuse.permute(0, 2, 1, 3, 4).contiguous().view(
            b,
            self.latent_dim,
            h,
            w,
        )

        delta = self.delta_mid(x_fuse)
        delta = self.delta_out(delta)

        gate = torch.sigmoid(self.gate(torch.cat([identity, delta], dim=1)))
        out = identity + gate * delta

        return out


@DETECTORS.register_module()
class BEVDetOCCMSBEVLHCRBEVAux(BEVDetOCCMSBEVBEVAux):
    """
    Full model:
        MS-BEV + LHCR-Full + BEVAux

    Inherited components:
        - MS-BEV from BEVDetOCCMSBEV
        - BEVAux from BEVDetOCCMSBEVBEVAux
        - Original BEVOCCHead2D is unchanged

    Inserted component:
        - LHCR after MS-BEV feature construction and before BEVOCCHead2D.
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

        img_feats[0] = self.lhcr_encoder(img_feats[0])

        if img_feats_is_tuple:
            img_feats = tuple(img_feats)

        return img_feats, pts_feats, depth
