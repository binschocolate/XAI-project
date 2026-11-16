#!/usr/bin/bash
#SBATCH -J xai-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 0-12
#SBATCH --array=0-5
#SBATCH -o logs/%x-%A_%a.out

# array index 0..5 -> (rho, seed)
#   0: rho0.95 seed0 | 1: rho0.95 seed1 | 2: rho0.95 seed2
#   3: rho0.5  seed0 | 4: rho0.5  seed1 | 5: rho0.5  seed2
#
# 환경변수로 바꿀 수 있는 값 (sbatch 앞에 붙여서 제출):
#   ARCH=resnet18|resnet50   (기본 resnet18)
#   EPOCHS=30                (기본 30)
#   OUT_SUB=models           (결과 하위 폴더, pilot은 pilot)
#   SKIP_EVAL=1              (학습만 하고 SRI 평가 생략)
# 결과 위치: /data/$USER/results/XAI-project/$OUT_SUB/<run>/
#
# 예) 본 학습:   sbatch scripts/train.sbatch
#     R50:       ARCH=resnet50 sbatch scripts/train.sbatch
#     pilot:     EPOCHS=50 OUT_SUB=pilot SKIP_EVAL=1 sbatch --array=0,3 scripts/train.sbatch
set -eo pipefail   # -u는 conda activate 스크립트와 충돌해서 뺌

ARCH=${ARCH:-resnet18}
EPOCHS=${EPOCHS:-30}
RESULTS=${RESULTS:-/data/$USER/results/XAI-project}
OUT_ROOT=$RESULTS/${OUT_SUB:-models}
SKIP_EVAL=${SKIP_EVAL:-0}

RHOS=(0.95 0.95 0.95 0.5 0.5 0.5)
SEEDS=(0 1 2 0 1 2)
RHO=${RHOS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd "$SLURM_SUBMIT_DIR"
source /data/$USER/anaconda3/etc/profile.d/conda.sh
conda activate xai
export TORCH_HOME=/data/$USER/cache/torch

echo "host=$(hostname) job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID arch=$ARCH rho=$RHO seed=$SEED epochs=$EPOCHS"
which python
nvidia-smi --query-gpu=name,memory.total --format=csv

source scripts/stage_data.sh

python src/train.py --arch "$ARCH" --rho "$RHO" --seed "$SEED" --epochs "$EPOCHS" \
  --data_dir "$DATA_DIR" --out_dir "$OUT_ROOT"

if [ "$SKIP_EVAL" != "1" ]; then
  RUN=$(python -c "import sys; sys.path.insert(0,'src'); from common import run_name; print(run_name('$ARCH', $RHO, $SEED))")
  python src/eval_sri.py --run_dir "$OUT_ROOT/$RUN" --data_dir "$DATA_DIR"
fi
echo "done"
exit 0
