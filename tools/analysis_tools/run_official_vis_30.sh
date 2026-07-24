#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
用法：
  bash tools/analysis_tools/run_official_vis_30.sh \
    --method "FlashOcc" \
    --config projects/configs/flashocc/model.py \
    --checkpoint path/to/model.pth \
    [--base-ann data/nuscenes/bevdetv2-nuscenes_infos_val.pkl] \
    [--num-frames 30] \
    [--random-seed 42] \
    [--gpus 0,1] \
    [--mask-mode camera] \
    [--zoom 0.08] \
    [--skip-inference] \
    [--force-inference] \
    [--overwrite]

说明：
  1. 固定 --num-frames 和 --random-seed，可让不同方法使用完全相同的帧。
  2. 每种方法只需更换 --method、--config、--checkpoint。
  3. 推理通过 dataset.evaluate(show_dir=...) 保存官方目录：
     prediction_root/scene_name/sample_token/pred.npz
  4. 每帧单独输出 Prediction、GT 和六张输入图像，不生成拼图。
EOF
}

METHOD=""
CONFIG=""
CHECKPOINT=""
BASE_ANN="data/nuscenes/bevdetv2-nuscenes_infos_val.pkl"
DATA_ROOT="data/nuscenes"
NUM_FRAMES=30
RANDOM_SEED=42
GPUS="0,1"
MASK_MODE="camera"
ZOOM=0.08
RENDER_WIDTH=2560
RENDER_HEIGHT=1440
OUT_ROOT="work_dirs/official_occ_vis"
RUN_NAME=""
ANN_CFG_KEY="auto"
SKIP_INFERENCE=0
FORCE_INFERENCE=0
OVERWRITE=0
HIDDEN=0
NO_VOXEL_EDGES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --base-ann|--ann-file) BASE_ANN="$2"; shift 2 ;;
    --data-root|--root-path) DATA_ROOT="$2"; shift 2 ;;
    --num-frames|--vis-frames) NUM_FRAMES="$2"; shift 2 ;;
    --random-seed|--seed) RANDOM_SEED="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --mask-mode) MASK_MODE="$2"; shift 2 ;;
    --zoom) ZOOM="$2"; shift 2 ;;
    --render-width) RENDER_WIDTH="$2"; shift 2 ;;
    --render-height) RENDER_HEIGHT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --ann-cfg-key) ANN_CFG_KEY="$2"; shift 2 ;;
    --skip-inference|--skip-eval) SKIP_INFERENCE=1; shift ;;
    --force-inference|--force-eval) FORCE_INFERENCE=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    --hidden) HIDDEN=1; shift ;;
    --no-voxel-edges) NO_VOXEL_EDGES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] 未知参数：$1"; usage; exit 1 ;;
  esac
done

if [[ -z "${METHOD}" || -z "${CONFIG}" || -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] --method、--config 和 --checkpoint 均为必填项。"
  usage
  exit 1
fi

for REQUIRED_FILE in \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  "${BASE_ANN}" \
  "tools/dist_test.sh" \
  "tools/analysis_tools/vis_occ.py" \
  "tools/analysis_tools/build_random_val_subset.py" \
  "tools/analysis_tools/verify_prediction_tree.py"; do
  if [[ ! -f "${REQUIRED_FILE}" ]]; then
    echo "[ERROR] 文件不存在：${REQUIRED_FILE}"
    exit 1
  fi
done

if ! [[ "${NUM_FRAMES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] --num-frames 必须是正整数：${NUM_FRAMES}"
  exit 1
fi

slugify() {
  echo "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+|_+$//g'
}

METHOD_SLUG="$(slugify "${METHOD}")"
CONFIG_STEM="$(basename "${CONFIG}")"
CONFIG_STEM="${CONFIG_STEM%.*}"
CHECKPOINT_STEM="$(basename "${CHECKPOINT}")"
CHECKPOINT_STEM="${CHECKPOINT_STEM%.*}"

if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="${METHOD_SLUG}__$(slugify "${CONFIG_STEM}")__$(slugify "${CHECKPOINT_STEM}")"
else
  RUN_NAME="$(slugify "${RUN_NAME}")"
fi

SUBSET_TAG="val_random_n${NUM_FRAMES}_seed${RANDOM_SEED}"
SUBSET_ANN="${OUT_ROOT}/subsets/${SUBSET_TAG}.pkl"
SUBSET_META="${OUT_ROOT}/subsets/${SUBSET_TAG}.json"
PRED_ROOT="${OUT_ROOT}/predictions/${RUN_NAME}/${SUBSET_TAG}"
FIG_ROOT="${OUT_ROOT}/figures/${RUN_NAME}/${SUBSET_TAG}"

mkdir -p \
  "${OUT_ROOT}/subsets" \
  "${OUT_ROOT}/predictions/${RUN_NAME}" \
  "${FIG_ROOT}"

SUBSET_ARGS=(
  --base-ann "${BASE_ANN}"
  --output-ann "${SUBSET_ANN}"
  --num-samples "${NUM_FRAMES}"
  --seed "${RANDOM_SEED}"
)
if [[ "${OVERWRITE}" -eq 1 ]]; then
  SUBSET_ARGS+=(--overwrite)
fi

python tools/analysis_tools/build_random_val_subset.py \
  "${SUBSET_ARGS[@]}"

if [[ "${ANN_CFG_KEY}" == "auto" ]]; then
  ANN_CFG_KEY=$(python - "${CONFIG}" <<'PY'
import sys
import mmcv

cfg = mmcv.Config.fromfile(sys.argv[1])

def find_ann(obj, path):
    if isinstance(obj, (dict, mmcv.ConfigDict)):
        if "ann_file" in obj:
            return path + ".ann_file"
        for key in ("dataset", "datasets"):
            if key in obj:
                result = find_ann(obj[key], path + "." + key)
                if result:
                    return result
        for key, value in obj.items():
            if key in ("dataset", "datasets"):
                continue
            result = find_ann(value, path + "." + str(key))
            if result:
                return result
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            result = find_ann(value, path + "." + str(index))
            if result:
                return result
    return None

key = find_ann(cfg.data.test, "data.test")
if not key:
    raise SystemExit("Cannot find ann_file under cfg.data.test")
print(key)
PY
  )
fi

echo "[INFO] 方法标识             : ${METHOD}"
echo "[INFO] 配置文件             : ${CONFIG}"
echo "[INFO] 权重文件             : ${CHECKPOINT}"
echo "[INFO] 随机帧数             : ${NUM_FRAMES}"
echo "[INFO] 随机种子             : ${RANDOM_SEED}"
echo "[INFO] 测试标注配置键       : ${ANN_CFG_KEY}"
echo "[INFO] 官方预测目录         : ${PRED_ROOT}"
echo "[INFO] 可视化输出目录       : ${FIG_ROOT}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
GPU_COUNT=${#GPU_ARRAY[@]}

NEED_INFERENCE=0
if [[ "${FORCE_INFERENCE}" -eq 1 ]]; then
  NEED_INFERENCE=1
elif [[ "${SKIP_INFERENCE}" -eq 1 ]]; then
  NEED_INFERENCE=0
elif [[ ! -d "${PRED_ROOT}" ]]; then
  NEED_INFERENCE=1
elif ! python tools/analysis_tools/verify_prediction_tree.py \
    --subset-ann "${SUBSET_ANN}" \
    --prediction-root "${PRED_ROOT}" >/dev/null 2>&1; then
  NEED_INFERENCE=1
fi

if [[ "${NEED_INFERENCE}" -eq 1 ]]; then
  rm -rf "${PRED_ROOT}"
  mkdir -p "${PRED_ROOT}"

  echo "[INFO] 开始推理并导出官方 pred.npz 目录。"
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  bash tools/dist_test.sh \
    "${CONFIG}" \
    "${CHECKPOINT}" \
    "${GPU_COUNT}" \
    --eval map \
    --eval-options "show_dir=${PRED_ROOT}" \
    --cfg-options "${ANN_CFG_KEY}=${SUBSET_ANN}"
elif [[ "${SKIP_INFERENCE}" -eq 1 ]]; then
  echo "[INFO] 跳过推理，使用已有官方预测目录。"
else
  echo "[INFO] 官方预测目录完整，直接进行可视化。"
fi

python tools/analysis_tools/verify_prediction_tree.py \
  --subset-ann "${SUBSET_ANN}" \
  --prediction-root "${PRED_ROOT}"

VIS_ARGS=(
  "${PRED_ROOT}"
  --root-path "${DATA_ROOT}"
  --info-path "${BASE_ANN}"
  --subset-meta "${SUBSET_META}"
  --save-path "${FIG_ROOT}"
  --vis-frames "${NUM_FRAMES}"
  --mask-mode "${MASK_MODE}"
  --render-width "${RENDER_WIDTH}"
  --render-height "${RENDER_HEIGHT}"
  --zoom "${ZOOM}"
)

if [[ "${OVERWRITE}" -eq 1 ]]; then
  VIS_ARGS+=(--overwrite)
fi
if [[ "${HIDDEN}" -eq 1 ]]; then
  VIS_ARGS+=(--hidden)
fi
if [[ "${NO_VOXEL_EDGES}" -eq 1 ]]; then
  VIS_ARGS+=(--no-voxel-edges)
fi

python tools/analysis_tools/vis_occ.py "${VIS_ARGS[@]}"

echo
echo "============================================================"
echo "FlashOcc官方前视角可视化完成"
echo "方法       : ${METHOD}"
echo "运行标识   : ${RUN_NAME}"
echo "帧数       : ${NUM_FRAMES}"
echo "随机种子   : ${RANDOM_SEED}"
echo "随机子集   : ${SUBSET_ANN}"
echo "预测目录   : ${PRED_ROOT}"
echo "图像目录   : ${FIG_ROOT}"
echo "============================================================"
