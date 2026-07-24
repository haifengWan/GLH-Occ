_base_ = ['./flashocc-r50-lhcr-z16-m1-24e.py']

# Practical latent-height ablation:
# only Z_l is changed; height_channels remains 32.
model = dict(
    lhcr_latent_height=8,
    lhcr_height_channels=32,
)

work_dir = './work_dirs/flashocc-r50-lhcr-z8-m1-24e'
