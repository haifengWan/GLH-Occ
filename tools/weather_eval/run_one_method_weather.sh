#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  bash tools/weather_eval/run_one_method_weather.sh \
    --method "FlashOcc" \
    --config projects/configs/flashocc/xxx.py \
    --checkpoint path/to/checkpoint.pth \
    [--backbone SwinB] \
    [--image-size 512x1408] \
    [--gpus 0,1] \
    [--scenarios rainy,night] \
    [--out-root work_dirs/scenario_eval] \
    [--force]

说明：
  - 一次只测试一个方法；
  - 默认依次测试 rainy 和 night；
  - 通过 --config 和 --checkpoint 指定任意模型，因此不在脚本中写死方法；
  - 默认使用 GPU 0,1，GPU 数量会根据 --gpus 自动计算；
  - 默认若该场景日志已完整存在，则跳过；加 --force 可重新测试。

示例：
  bash tools/weather_eval/run_one_method_weather.sh \
    --method "BEVDetOcc" \
    --config projects/configs/bevdet_occ/bevdet-occ-stbase-4d-stereo-512x1408.py \
    --checkpoint ckpts/bevdet-stbase-4d-stereo-512x1408-cbgs.pth

  bash tools/weather_eval/run_one_method_weather.sh \
    --method "FlashOcc" \
    --config projects/configs/flashocc/flashocc-stbase-4d-stereo-512x1408-m3-test-compat.py \
    --checkpoint ckpts/flashocc-stbase-4d-stereo-512x1408.pth

  bash tools/weather_eval/run_one_method_weather.sh \
    --method "MSBEV-BEVAux-M1" \
    --config projects/configs/flashocc/flashocc-r50-msbev-bevaux-m1-24e.py \
    --checkpoint work_dirs/flashocc-r50-msbev-bevaux-m1-24e/epoch_24_ema.pth \
    --backbone ResNet50 \
    --image-size 256x704 \
    --gpus 0
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
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)
      METHOD="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --backbone)
      BACKBONE="$2"
      shift 2
      ;;
    --image-size)
      IMAGE_SIZE="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --scenarios)
      SCENARIOS="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] 未知参数：$1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${METHOD}" || -z "${CONFIG}" || -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] --method、--config、--checkpoint 均为必填参数。"
  usage
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] 配置文件不存在：${CONFIG}"
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] checkpoint 不存在：${CHECKPOINT}"
  exit 1
fi

if [[ ! -f "tools/dist_test.sh" ]]; then
  echo "[ERROR] 请从 FlashOcc 工程根目录运行本脚本。"
  exit 1
fi

if [[ ! -f "tools/weather_eval/parse_occ_eval_log.py" ]]; then
  echo "[ERROR] 缺少日志解析脚本：tools/weather_eval/parse_occ_eval_log.py"
  exit 1
fi

METHOD_SLUG="$(echo "${METHOD}" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+|_+$//g')"
GPU_NUM="$(awk -F',' '{print NF}' <<< "${GPUS}")"

mkdir -p \
  "${OUT_ROOT}/outputs" \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/results"

METHOD_CSV="${OUT_ROOT}/results/${METHOD_SLUG}_weather_results.csv"

# 本次执行重新生成该方法的CSV，避免重复追加。
rm -f "${METHOD_CSV}"

IFS=',' read -r -a SCENARIO_ARRAY <<< "${SCENARIOS}"

echo "============================================================"
echo "方法名称       : ${METHOD}"
echo "方法标识       : ${METHOD_SLUG}"
echo "配置文件       : ${CONFIG}"
echo "Checkpoint     : ${CHECKPOINT}"
echo "Backbone       : ${BACKBONE}"
echo "图像尺寸       : ${IMAGE_SIZE}"
echo "可见GPU        : ${GPUS}"
echo "GPU数量        : ${GPU_NUM}"
echo "测试场景       : ${SCENARIOS}"
echo "输出目录       : ${OUT_ROOT}"
echo "============================================================"

for SCENARIO in "${SCENARIO_ARRAY[@]}"; do
  SCENARIO="$(echo "${SCENARIO}" | xargs)"

  case "${SCENARIO}" in
    rainy|night)
      ;;
    *)
      echo "[ERROR] 不支持的场景：${SCENARIO}，仅支持 rainy 或 night。"
      exit 1
      ;;
  esac

  ANN_FILE="data/nuscenes/bevdetv2-nuscenes_infos_val_${SCENARIO}.pkl"
  OUTPUT_FILE="${OUT_ROOT}/outputs/${METHOD_SLUG}_${SCENARIO}_outputs.pkl"
  LOG_FILE="${OUT_ROOT}/logs/${METHOD_SLUG}_${SCENARIO}.log"

  if [[ ! -f "${ANN_FILE}" ]]; then
    echo "[ERROR] 子集标注文件不存在：${ANN_FILE}"
    exit 1
  fi

  EXPECTED_SAMPLES="$(
    python - "${ANN_FILE}" <<'PY'
import sys
import mmcv

path = sys.argv[1]
data = mmcv.load(path)
if isinstance(data, dict):
    infos = data.get("infos", data.get("data_list"))
else:
    infos = data

if infos is None:
    raise RuntimeError(f"Cannot locate infos/data_list in {path}")

print(len(infos))
PY
  )"

  if [[ "${FORCE}" -eq 0 && -f "${LOG_FILE}" ]]; then
    if grep -q "===> mIoU of ${EXPECTED_SAMPLES} samples:" "${LOG_FILE}"; then
      echo
      echo "[SKIP] ${METHOD} / ${SCENARIO} 已有完整日志：${LOG_FILE}"
      python tools/weather_eval/parse_occ_eval_log.py \
        --log "${LOG_FILE}" \
        --method "${METHOD}" \
        --backbone "${BACKBONE}" \
        --image-size "${IMAGE_SIZE}" \
        --scenario "${SCENARIO}" \
        --output "${METHOD_CSV}"
      continue
    fi
  fi

  echo
  echo "------------------------------------------------------------"
  echo "开始测试"
  echo "方法           : ${METHOD}"
  echo "场景           : ${SCENARIO}"
  echo "预期样本数     : ${EXPECTED_SAMPLES}"
  echo "标注文件       : ${ANN_FILE}"
  echo "输出文件       : ${OUTPUT_FILE}"
  echo "日志文件       : ${LOG_FILE}"
  echo "------------------------------------------------------------"

  CUDA_VISIBLE_DEVICES="${GPUS}" \
  bash tools/dist_test.sh \
    "${CONFIG}" \
    "${CHECKPOINT}" \
    "${GPU_NUM}" \
    --out "${OUTPUT_FILE}" \
    --eval map \
    --cfg-options data.test.ann_file="${ANN_FILE}" \
    2>&1 | tee "${LOG_FILE}"

  if ! grep -q "===> mIoU of ${EXPECTED_SAMPLES} samples:" "${LOG_FILE}"; then
    echo "[ERROR] 日志中未找到预期的 mIoU 结果。"
    echo "        预期样本数：${EXPECTED_SAMPLES}"
    echo "        日志文件：${LOG_FILE}"
    exit 1
  fi

  CLASS_COUNT="$(grep -c "IoU =" "${LOG_FILE}" || true)"
  if [[ "${CLASS_COUNT}" -ne 17 ]]; then
    echo "[ERROR] 检测到 ${CLASS_COUNT} 个类别IoU，预期为17个。"
    exit 1
  fi

  python tools/weather_eval/parse_occ_eval_log.py \
    --log "${LOG_FILE}" \
    --method "${METHOD}" \
    --backbone "${BACKBONE}" \
    --image-size "${IMAGE_SIZE}" \
    --scenario "${SCENARIO}" \
    --output "${METHOD_CSV}"

  echo "[DONE] ${METHOD} / ${SCENARIO}"
done

echo
echo "============================================================"
echo "该方法的雨天/夜间测试已完成。"
echo "日志目录 : ${OUT_ROOT}/logs"
echo "输出目录 : ${OUT_ROOT}/outputs"
echo "结果CSV  : ${METHOD_CSV}"
echo "============================================================"
