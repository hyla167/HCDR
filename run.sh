#!/bin/bash
# End-to-end pipeline: train each (method, dataset, mem_size) config, then run
# the linear-probe evaluation on the checkpoint that was just produced.
#
# Both methods now share a single entrypoint pair:
#     python train.py    --method {cclis|hcdr} ...
#     python evaluate.py  --method {cclis|hcdr} ...
# HCDR runs are tagged with an "hcdr" substring in their model_name (see
# config.py), so the grep-based directory selection below still distinguishes
# the two methods under a shared save root.
export CUDA_VISIBLE_DEVICES="2"

METHODS=("hcdr")
DATASETS=("tiny-imagenet")
MEM_SIZES=(1000 2000 3000 4000)

TRAIN_START_EPOCH=500
TRAIN_EPOCHS=50

set -e

for METHOD in "${METHODS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for MEM in "${MEM_SIZES[@]}"; do

      echo "================================================================================"
      echo "Running Pipeline: ${METHOD^^} | Dataset: ${DATASET} | Mem Size: ${MEM}"
      echo "================================================================================"

      echo ">>> [Phase A] Starting Training for ${METHOD^^}..."
      python train.py \
        --method "${METHOD}" \
        --dataset "${DATASET}" \
        --mem_size "${MEM}" \
        --start_epoch "${TRAIN_START_EPOCH}" \
        --epochs "${TRAIN_EPOCHS}"

      BASE_SAVE_DIR="./save_weight_${MEM}_"

      # ls -td -> newest run first; grep filters by the "hcdr" tag.
      if [ "$METHOD" = "cclis" ]; then
        CKPT_DIR=$(ls -td ${BASE_SAVE_DIR}/${DATASET}_models/${DATASET}_* 2>/dev/null | grep -v "hcdr" | head -n 1)
        LOG_DIR=$(ls -td ${BASE_SAVE_DIR}/logs/${DATASET}_* 2>/dev/null | grep -v "hcdr" | head -n 1)
      elif [ "$METHOD" = "hcdr" ]; then
        CKPT_DIR=$(ls -td ${BASE_SAVE_DIR}/${DATASET}_models/${DATASET}_*hcdr* 2>/dev/null | head -n 1)
        LOG_DIR=$(ls -td ${BASE_SAVE_DIR}/logs/${DATASET}_*hcdr* 2>/dev/null | head -n 1)
      else
        echo "Unknown method: $METHOD"; exit 1
      fi

      if [ -z "$CKPT_DIR" ] || [ ! -d "$CKPT_DIR" ] || [ -z "$LOG_DIR" ] || [ ! -d "$LOG_DIR" ]; then
        echo "Error: Could not find recently created checkpoint or log directories!"
        echo "Looked for CKPT in: ${BASE_SAVE_DIR}/${DATASET}_models/"
        exit 1
      fi

      echo ">>> [Phase B] Starting Linear Evaluation..."
      echo "    -> Using CKPT: $CKPT_DIR"
      echo "    -> Using LOGS: $LOG_DIR"

      python evaluate.py \
        --method "${METHOD}" \
        --dataset "${DATASET}" \
        --ckpt "${CKPT_DIR}" \
        --logpt "${LOG_DIR}"

      echo ">>> Finished ${METHOD^^} on ${DATASET} (Mem Size: ${MEM})"
      echo ""
    done
  done
done

echo "================================================================================"
echo "All experiments completed successfully!"
echo "================================================================================"