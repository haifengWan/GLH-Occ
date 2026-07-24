# FlashOcc official front-overlook visualization: 30 frames and multiple methods

Install all four files into `tools/analysis_tools/`:

```bash
cd ~/projects/FlashOCC

cp /下载位置/vis_occ.py tools/analysis_tools/
cp /下载位置/build_random_val_subset.py tools/analysis_tools/
cp /下载位置/verify_prediction_tree.py tools/analysis_tools/
cp /下载位置/run_official_vis_30.sh tools/analysis_tools/

chmod +x tools/analysis_tools/*.py
chmod +x tools/analysis_tools/run_official_vis_30.sh
```

The wrapper:

1. deterministically selects 30 validation samples;
2. overrides the method config's test `ann_file`;
3. runs the selected config and checkpoint;
4. asks `dataset.evaluate()` to save official
   `scene_name/sample_token/pred.npz`;
5. renders Prediction, GT, and six input images separately.

Keep these fixed across methods:

```text
--base-ann
--num-frames 30
--random-seed 42
```

Change these per method:

```text
--method
--config
--checkpoint
```
