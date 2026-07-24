import torch
from pathlib import Path

src_path = Path('ckpts/flashocc-r50-256x704.pth')
dst_path = Path('ckpts/flashocc-r50-256x704-scgl-g4-init.pth')

G = 4
D = 88
C = 64
old_out = D + C
new_out = G * D + C

ckpt = torch.load(str(src_path), map_location='cpu')
sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

new_sd = sd.copy()

converted = []

for k, v in list(sd.items()):
    if not k.endswith('depth_net.weight'):
        continue

    if v.ndim != 4:
        continue

    if v.shape[0] != old_out:
        continue

    bias_key = k.replace('.weight', '.bias')
    if bias_key not in sd:
        continue

    b = sd[bias_key]
    if b.shape[0] != old_out:
        continue

    depth_w = v[:D].clone()
    feat_w = v[D:D + C].clone()

    depth_b = b[:D].clone()
    feat_b = b[D:D + C].clone()

    # New order:
    # [depth_group_0, depth_group_1, depth_group_2, depth_group_3, tran_feat]
    new_w = torch.cat([depth_w for _ in range(G)] + [feat_w], dim=0)
    new_b = torch.cat([depth_b for _ in range(G)] + [feat_b], dim=0)

    assert new_w.shape[0] == new_out
    assert new_b.shape[0] == new_out

    new_sd[k] = new_w
    new_sd[bias_key] = new_b

    converted.append((k, tuple(v.shape), tuple(new_w.shape)))

if 'state_dict' in ckpt:
    ckpt['state_dict'] = new_sd
else:
    ckpt = new_sd

torch.save(ckpt, str(dst_path))

print('[OK] saved:', dst_path)
print('[Converted depth_net keys]')
for item in converted:
    print(item)

if not converted:
    print('[WARN] no depth_net keys converted. Check checkpoint key names.')
