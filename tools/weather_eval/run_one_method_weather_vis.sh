#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  bash tools/weather_eval/run_one_method_weather_vis.sh \
    --method "FlashOcc" \
    --config projects/configs/flashocc/xxx.py \
    --checkpoint path/to/checkpoint.pth \
    [--backbone SwinB] \
    [--image-size 512x1408] \
    [--gpus 0,1] \
    [--scenarios rainy,night] \
    [--out-root work_dirs/scenario_eval] \
    [--vis-root work_dirs/scenario_vis] \
    [--num-scenes 20] \
    [--selection unique_scene] \
    [--mask-mode camera] \
    [--max-points 0] \
    [--force-eval] \
    [--skip-eval] \
    [--overwrite] \
    [--draw-gt | --no-draw-gt] \
    [--camera-view-mode occformer|pinhole|directional] \
    [--allow-camera-fallback]

说明：
  1. 复用 run_one_method_weather.sh 生成的 *_outputs.pkl，不重复推理。
  2. 默认 rainy、night 各选择 20 个代表帧。
  3. 更换方法或权重时只需更换命令行参数；同名方法换权重时建议加 --force-eval。
EOF
}

METHOD=""
CONFIG=""
CHECKPOINT=""
BACKBONE="SwinB"
IMAGE_SIZE="512x1408"
GPUS="0,1"
SCENARIOS="rainy,night"
OUT_ROOT="work_dirs/scenario_eval"
VIS_ROOT="work_dirs/scenario_vis"
NUM_SCENES=20
SELECTION="unique_scene"
MASK_MODE="camera"
MAX_POINTS=0
FORCE_EVAL=0
SKIP_EVAL=0
OVERWRITE=0
SHOW_ALL_VOXELS=0
RENDERER="open3d"
RENDER_WIDTH=2560
RENDER_HEIGHT=1440
OPEN3D_HIDDEN=0
NO_VOXEL_EDGES=0
EDGE_VOXEL_LIMIT=150000
CAMERA_ZOOM=0.10
OVERVIEW_ZOOM=0.08
BEV_ZOOM=0.15
DRAW_GT=1
CAMERA_VIEW_MODE="occformer"
STRICT_CAMERA_CALIBRATION=1
OCCFORMER_FOCAL_DISTANCE=0.0055
OCCFORMER_VIEW_ANGLE=35.0
OCCFORMER_BACK_LEFT_VIEW_ANGLE=60.0
OCCFORMER_CAMERA_OFFSET="0,0,0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --backbone) BACKBONE="$2"; shift 2 ;;
    --image-size) IMAGE_SIZE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --scenarios) SCENARIOS="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --vis-root) VIS_ROOT="$2"; shift 2 ;;
    --num-scenes) NUM_SCENES="$2"; shift 2 ;;
    --selection) SELECTION="$2"; shift 2 ;;
    --mask-mode) MASK_MODE="$2"; shift 2 ;;
    --max-points) MAX_POINTS="$2"; shift 2 ;;
    --force-eval) FORCE_EVAL=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    --show-all-voxels) SHOW_ALL_VOXELS=1; shift ;;
    --renderer) RENDERER="$2"; shift 2 ;;
    --render-width) RENDER_WIDTH="$2"; shift 2 ;;
    --render-height) RENDER_HEIGHT="$2"; shift 2 ;;
    --open3d-hidden) OPEN3D_HIDDEN=1; shift ;;
    --no-voxel-edges) NO_VOXEL_EDGES=1; shift ;;
    --edge-voxel-limit) EDGE_VOXEL_LIMIT="$2"; shift 2 ;;
    --camera-zoom) CAMERA_ZOOM="$2"; shift 2 ;;
    --overview-zoom) OVERVIEW_ZOOM="$2"; shift 2 ;;
    --bev-zoom) BEV_ZOOM="$2"; shift 2 ;;
    --draw-gt) DRAW_GT=1; shift ;;
    --camera-view-mode) CAMERA_VIEW_MODE="$2"; shift 2 ;;
    --occformer-focal-distance) OCCFORMER_FOCAL_DISTANCE="$2"; shift 2 ;;
    --occformer-view-angle) OCCFORMER_VIEW_ANGLE="$2"; shift 2 ;;
    --occformer-back-left-view-angle) OCCFORMER_BACK_LEFT_VIEW_ANGLE="$2"; shift 2 ;;
    --occformer-camera-offset) OCCFORMER_CAMERA_OFFSET="$2"; shift 2 ;;
    --allow-camera-fallback) STRICT_CAMERA_CALIBRATION=0; shift ;;
    --no-draw-gt) DRAW_GT=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] 未知参数：$1"; usage; exit 1 ;;
  esac
done

if [[ -z "${METHOD}" || -z "${CONFIG}" || -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] --method、--config、--checkpoint 均为必填参数。"
  usage
  exit 1
fi

for REQUIRED_FILE in \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  "tools/weather_eval/run_one_method_weather.sh" \
  "tools/weather_eval/visualize_occ_weather.py"; do
  if [[ ! -f "${REQUIRED_FILE}" ]]; then
    echo "[ERROR] 文件不存在：${REQUIRED_FILE}"
    exit 1
  fi
done

METHOD_SLUG="$(echo "${METHOD}" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+|_+$//g')"
IFS=',' read -r -a SCENARIO_ARRAY <<< "${SCENARIOS}"

NEED_EVAL=0
for SCENARIO_RAW in "${SCENARIO_ARRAY[@]}"; do
  SCENARIO="$(echo "${SCENARIO_RAW}" | xargs)"
  case "${SCENARIO}" in
    rainy|night) ;;
    *) echo "[ERROR] 不支持的场景：${SCENARIO}"; exit 1 ;;
  esac
  PRED_FILE="${OUT_ROOT}/outputs/${METHOD_SLUG}_${SCENARIO}_outputs.pkl"
  if [[ ! -f "${PRED_FILE}" ]]; then
    NEED_EVAL=1
  fi
done

if [[ "${SKIP_EVAL}" -eq 0 && ( "${NEED_EVAL}" -eq 1 || "${FORCE_EVAL}" -eq 1 ) ]]; then
  EVAL_ARGS=(
    --method "${METHOD}"
    --config "${CONFIG}"
    --checkpoint "${CHECKPOINT}"
    --backbone "${BACKBONE}"
    --image-size "${IMAGE_SIZE}"
    --gpus "${GPUS}"
    --scenarios "${SCENARIOS}"
    --out-root "${OUT_ROOT}"
  )
  if [[ "${FORCE_EVAL}" -eq 1 ]]; then
    EVAL_ARGS+=(--force)
  fi
  bash tools/weather_eval/run_one_method_weather.sh "${EVAL_ARGS[@]}"
elif [[ "${SKIP_EVAL}" -eq 1 ]]; then
  echo "[INFO] 跳过推理，直接读取已有预测文件。"
else
  echo "[INFO] 预测文件已存在，直接开始离线可视化。"
fi

for SCENARIO_RAW in "${SCENARIO_ARRAY[@]}"; do
  SCENARIO="$(echo "${SCENARIO_RAW}" | xargs)"
  ANN_FILE="data/nuscenes/bevdetv2-nuscenes_infos_val_${SCENARIO}.pkl"
  PRED_FILE="${OUT_ROOT}/outputs/${METHOD_SLUG}_${SCENARIO}_outputs.pkl"
  VIS_DIR="${VIS_ROOT}/${METHOD_SLUG}/${SCENARIO}"

  if [[ ! -f "${ANN_FILE}" ]]; then
    echo "[ERROR] 子集标注不存在：${ANN_FILE}"
    exit 1
  fi
  if [[ ! -f "${PRED_FILE}" ]]; then
    echo "[ERROR] 预测文件不存在：${PRED_FILE}"
    echo "        请移除 --skip-eval，或先完成天气子集测试。"
    exit 1
  fi

  VIS_ARGS=(
    --config "${CONFIG}"
    --ann-file "${ANN_FILE}"
    --pred-file "${PRED_FILE}"
    --output-dir "${VIS_DIR}"
    --method "${METHOD}"
    --scenario "${SCENARIO}"
    --num-scenes "${NUM_SCENES}"
    --selection "${SELECTION}"
    --mask-mode "${MASK_MODE}"
    --max-points "${MAX_POINTS}"
    --renderer "${RENDERER}"
    --render-width "${RENDER_WIDTH}"
    --render-height "${RENDER_HEIGHT}"
    --edge-voxel-limit "${EDGE_VOXEL_LIMIT}"
    --camera-zoom "${CAMERA_ZOOM}"
    --overview-zoom "${OVERVIEW_ZOOM}"
    --bev-zoom "${BEV_ZOOM}"
    --camera-view-mode "${CAMERA_VIEW_MODE}"
    --occformer-focal-distance "${OCCFORMER_FOCAL_DISTANCE}"
    --occformer-view-angle "${OCCFORMER_VIEW_ANGLE}"
    --occformer-back-left-view-angle "${OCCFORMER_BACK_LEFT_VIEW_ANGLE}"
    --occformer-camera-offset "${OCCFORMER_CAMERA_OFFSET}"
  )
  if [[ "${OVERWRITE}" -eq 1 ]]; then
    VIS_ARGS+=(--overwrite)
  fi
  if [[ "${SHOW_ALL_VOXELS}" -eq 1 ]]; then
    VIS_ARGS+=(--show-all-voxels)
  fi
  if [[ "${OPEN3D_HIDDEN}" -eq 1 ]]; then
    VIS_ARGS+=(--open3d-hidden)
  fi
  if [[ "${NO_VOXEL_EDGES}" -eq 1 ]]; then
    VIS_ARGS+=(--no-voxel-edges)
  fi
  if [[ "${STRICT_CAMERA_CALIBRATION}" -eq 1 ]]; then
    VIS_ARGS+=(--strict-camera-calibration)
  fi
  if [[ "${DRAW_GT}" -eq 1 ]]; then
    VIS_ARGS+=(--draw-gt)
  else
    VIS_ARGS+=(--no-draw-gt)
  fi

  echo "[INFO] Effective BEV_ZOOM=${BEV_ZOOM}"
  echo "[INFO] CAMERA_VIEW_MODE=${CAMERA_VIEW_MODE}"
  echo "[INFO] STRICT_CAMERA_CALIBRATION=${STRICT_CAMERA_CALIBRATION}"
  echo "[INFO] OCCFORMER_CAMERA_OFFSET=${OCCFORMER_CAMERA_OFFSET}"
  python tools/weather_eval/visualize_occ_weather.py "${VIS_ARGS[@]}"
done

echo
echo "============================================================"
echo "天气场景可视化完成"
echo "方法      : ${METHOD}"
echo "场景      : ${SCENARIOS}"
echo "每场景数量: ${NUM_SCENES}"
echo "GT对比行  : ${DRAW_GT}"
echo "BEV zoom  : ${BEV_ZOOM}"
echo "相机视图  : ${CAMERA_VIEW_MODE}"
echo "严格标定  : ${STRICT_CAMERA_CALIBRATION}"
echo "OccFormer offset: ${OCCFORMER_CAMERA_OFFSET}"
echo "输出目录  : ${VIS_ROOT}/${METHOD_SLUG}"
echo "============================================================"
