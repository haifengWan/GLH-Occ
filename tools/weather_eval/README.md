# FlashOcc 天气场景可视化（V3：更清晰的占据图）

## 1. 放置脚本

复制到 FlashOCC 工程：

```text
tools/weather_eval/visualize_occ_weather.py
tools/weather_eval/run_one_method_weather_vis.sh
```

并赋权：

```bash
cd ~/projects/FlashOCC
chmod +x tools/weather_eval/visualize_occ_weather.py
chmod +x tools/weather_eval/run_one_method_weather_vis.sh
```

## 2. 仅重排版，不重新推理

如果已有：

```text
work_dirs/scenario_eval/outputs/FlashOcc_rainy_outputs.pkl
work_dirs/scenario_eval/outputs/FlashOcc_night_outputs.pkl
```

直接运行：

```bash
cd ~/projects/FlashOCC
conda activate FlashOcc

bash tools/weather_eval/run_one_method_weather_vis.sh \
  --method "FlashOcc" \
  --config projects/configs/flashocc/flashocc-stbase-4d-stereo-512x1408-m3-test-compat.py \
  --checkpoint ckpts/flashocc-stbase-4d-stereo-512x1408.pth \
  --backbone SwinB \
  --image-size 512x1408 \
  --gpus 0,1 \
  --num-scenes 10 \
  --skip-eval \
  --overwrite
```

## 3. 与上一版相比的改进

新版占据图更清晰，主要原因有：

1. 默认 `--max-points 120000`，不再只保留 30000 个点；
2. 仍然只画表面体素，因此保留结构感且不会被内部体素遮挡；
3. 自动围绕有效占据区域缩放，不再总是使用整个 `[-40, 40] × [-40, 40]` 画布；
4. 渲染前按观察方向进行深度排序，近处结构更稳定；
5. 提高了主视图和六个小视图的 marker size；
6. 使用略高对比度的配色，减少 “manmade 过白导致看不清” 的问题。

## 4. 若想更密集

可以继续提高：

```bash
--max-points 180000
```

例如：

```bash
bash tools/weather_eval/run_one_method_weather_vis.sh \
  --method "FlashOcc" \
  --config projects/configs/flashocc/flashocc-stbase-4d-stereo-512x1408-m3-test-compat.py \
  --checkpoint ckpts/flashocc-stbase-4d-stereo-512x1408.pth \
  --backbone SwinB \
  --image-size 512x1408 \
  --gpus 0,1 \
  --num-scenes 10 \
  --skip-eval \
  --overwrite \
  --max-points 180000
```

## 5. 若想把所有表面体素都画出来

```bash
--show-all-voxels --max-points 0
```

这会更接近“密实体素块”的风格，但渲染会明显更慢。

## 6. 独立图像输出

每个样本目录仍然包含：

```text
overall.png
panels/
├── camera_grid.png
├── occupancy_views_grid.png
├── front_overlook.png
├── bev.png
├── legend.png
├── rgb_CAM_FRONT_LEFT.png
├── ...
├── occ_CAM_FRONT_LEFT.png
└── ...
```

因此如果你觉得最终总图还不够满意，可以直接使用 `panels` 目录中的独立高清图在论文或 PPT 中自行拼接。
