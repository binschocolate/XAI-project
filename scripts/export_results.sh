#!/usr/bin/env bash
# Seraph 결과 폴더에서 "작은 결과 파일"만 repo의 results/<arch>/ 로 복사한다.
# checkpoint(final.pt)는 복사하지 않는다 (git 제외, Seraph에만 보관).
# cp/grep만 쓰므로 master 노드에서 실행해도 된다.
#
# 사용: bash scripts/export_results.sh resnet18
set -eo pipefail

ARCH=${1:?usage: bash scripts/export_results.sh <resnet18|resnet50>}
SRC=${RESULTS:-/data/$USER/results/XAI-project}
DST=results/$ARCH

n=0
for d in "$SRC"/models/${ARCH}_rho*_seed*; do
  [ -d "$d" ] || continue
  run=$(basename "$d")
  mkdir -p "$DST/runs/$run"
  for f in config.json train_log.csv metrics.json test_preds.csv; do
    [ -f "$d/$f" ] && cp "$d/$f" "$DST/runs/$run/"
  done
  n=$((n + 1))
done

for d in "$SRC"/pilot/${ARCH}_rho*_seed*; do
  [ -d "$d" ] || continue
  run=$(basename "$d")
  mkdir -p "$DST/pilot/$run"
  for f in config.json train_log.csv; do
    [ -f "$d/$f" ] && cp "$d/$f" "$DST/pilot/$run/"
  done
done

# 전체 요약 csv에서 해당 arch 행만 추출
if [ -f "$SRC/sri_runs.csv" ]; then
  { head -1 "$SRC/sri_runs.csv"; grep "^${ARCH}_" "$SRC/sri_runs.csv" || true; } > "$DST/sri_runs.csv"
fi

echo "[export] $n runs -> $DST"
find "$DST" -type f | sort
