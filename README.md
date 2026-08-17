<div align="center">

# GLH-Occ: Geometry-Adaptive Lifting and Latent Height Context Reasoning for Efficient Semantic Occupancy Prediction

**An efficient vision-only semantic occupancy framework with adaptive geometry and height-aware BEV reasoning**

</div>

> [!NOTE]
> The manuscript has been completed and is being prepared for submission. The paper, final training configuration, pretrained checkpoints, and complete experiment logs will be released when they are ready for public distribution.

## Framework

<p align="center">
  <img src="https://haifengwan.github.io/GLH-Occ/assets/framework.jpg" alt="Overview of the GLH-Occ framework" width="100%">
</p>

GLH-Occ improves 3D scene modeling at both the image-to-BEV transformation and BEV representation stages. It preserves efficient BEV inference while introducing geometry-adaptive lifting, multi-scale feature aggregation, latent height reasoning, and training-time semantic regularization.

## Highlights

- **Geometry-adaptive lifting:** learns bounded, group-specific depth residuals on top of a shared depth prior.
- **Latent height reasoning:** captures vertical dependencies, local spatial relations, and global scene context before voxel decoding.
- **Competitive accuracy:** reaches **45.1% mIoU**, outperforming FlashOCC by **1.7 percentage points** in our current evaluation.
- **Robust qualitative behavior:** preserves coherent occupancy predictions in challenging rainy and nighttime scenes.

## Method

GLH-Occ contains four main components:

1. **Residual Semantic Channel-Grouped Lifting (R-SCGL).**

   R-SCGL retains a shared base depth prior and predicts bounded depth residuals for different feature groups. Image features with heterogeneous semantic responses can therefore use group-specific lifting distributions without discarding the stable shared geometry.

2. **Multi-Scale BEV Feature Fusion.**

   Features from multiple image scales are lifted and fused in BEV space, combining fine structural details with broader scene context.

3. **Latent Height Context Reasoning (LHCR).**

   LHCR reorganizes BEV channels into latent height representations and models dependencies along the height dimension, local spatial interactions, and global prototype context. The resulting BEV features contain richer vertical and structural information.

4. **BEV Auxiliary Supervision.**

   A training-only auxiliary head projects occupancy ground truth into BEV and regularizes intermediate features, improving their semantic consistency without adding inference-time cost.

## Release Status

- [x] Core model implementation and experimental configurations
- [x] Training, evaluation, and visualization utilities
- [ ] Paper

## Installation

The codebase is developed on top of [FlashOCC](https://github.com/Yzichen/FlashOCC) and follows its OpenMMLab environment. The reference environment uses Linux, Python 3.8, PyTorch 1.10, CUDA 11.1, MMCV 1.5.3, MMDetection 2.25.1, MMSegmentation 0.25.0, and MMDetection3D v1.0.0rc4.

### 1. Create the environment

```bash
conda create -n glhocc python=3.8.5 -y
conda activate glhocc

pip install torch==1.10.0+cu111 \
    torchvision==0.11.0+cu111 \
    torchaudio==0.10.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.5.3
pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0
pip install numpy==1.23.5 numba==0.53.0 nuscenes-devkit \
    plyfile scikit-image tensorboard setuptools==59.5.0 yapf==0.40.1
```

CUDA and MMCV builds are platform-dependent. If the commands above do not match your CUDA toolchain, follow the [FlashOCC environment guide](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md) and install the corresponding MMCV build.

### 2. Install GLH-Occ and MMDetection3D

```bash
git clone https://github.com/haifengWan/GLH-Occ.git
cd GLH-Occ

git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc4
pip install -v -e .

cd ../projects
pip install -v -e .
cd ..
```

## Data Preparation

Download the nuScenes train/validation set and occupancy ground truth following the [FlashOCC data preparation guide](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md). Arrange the files as follows:

```text
GLH-Occ/
└── data/
    └── nuscenes/
        ├── v1.0-trainval/
        ├── samples/
        ├── sweeps/
        ├── gts/
        ├── bevdetv2-nuscenes_infos_train.pkl
        └── bevdetv2-nuscenes_infos_val.pkl
```

Generate the BEVDet-style nuScenes information files with:

```bash
python tools/create_data_bevdet.py
```

The current configurations initialize from the FlashOCC R50 checkpoint. Download the corresponding checkpoint from the [FlashOCC model table](https://github.com/Yzichen/FlashOCC#main-results) and place it at:

```text
ckpts/flashocc-r50-256x704.pth
```

## Training

Experimental configurations are provided in [`projects/configs/GLH-Occ/`](projects/configs/GLH-Occ/). For example:

```bash
CONFIG=projects/configs/GLH-Occ/flashocc-r50-msbev-lhcr-full-bevaux-m1-24e.py

# Single GPU
python tools/train.py ${CONFIG}

# Multiple GPUs
bash tools/dist_train.sh ${CONFIG} 4
```

The final all-module configuration used for the submission will be added with the public model release.

## Evaluation

```bash
CONFIG=projects/configs/GLH-Occ/flashocc-r50-msbev-lhcr-full-bevaux-m1-24e.py
CHECKPOINT=path/to/glh_occ_checkpoint.pth

# Semantic occupancy evaluation
bash tools/dist_test.sh ${CONFIG} ${CHECKPOINT} 4 --eval mIoU

# RayIoU evaluation
bash tools/dist_test.sh ${CONFIG} ${CHECKPOINT} 4 --eval ray-iou
```

Single-GPU evaluation is also supported:

```bash
python tools/test.py ${CONFIG} ${CHECKPOINT} --eval mIoU
```

## Qualitative Results

The example below shows GLH-Occ predictions in a challenging nighttime scene. The model preserves the road layout and surrounding vertical structures despite low illumination and strong headlights.

<p align="center">
  <img src="https://haifengwan.github.io/GLH-Occ/assets/night_qualitative.jpg" alt="GLH-Occ qualitative results in a nighttime scene" width="100%">
</p>

## Acknowledgements

This project is based on [FlashOCC](https://github.com/Yzichen/FlashOCC) and builds upon the OpenMMLab ecosystem, including [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), as well as the [BEVDet](https://github.com/HuangJunJie2017/BEVDet) codebase. We thank the authors and maintainers for making their work publicly available.

## Citation

Citation information for GLH-Occ will be added after the manuscript metadata becomes public. In the meantime, please also consider citing [FlashOCC](https://arxiv.org/abs/2311.12058), on which this implementation is based.

## License

GLH-Occ licensing information will be provided before the public release. Third-party code and dependencies remain subject to their original licenses.
