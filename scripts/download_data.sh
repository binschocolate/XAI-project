#!/usr/bin/env bash
# Waterbirds tar.gz를 NAS에 한 번만 받아둔다 (압축은 여기서 풀지 않음).
#   bash scripts/download_data.sh      (srun으로 받은 GPU 노드에서)
set -euo pipefail

TAR_DIR=/data/$USER/datasets
TAR=$TAR_DIR/waterbird_complete95_forest2water2.tar.gz
URL=https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz

mkdir -p "$TAR_DIR"
if [ -f "$TAR" ]; then
  echo "already exists: $TAR"
else
  wget -c -O "$TAR" "$URL"
fi
ls -lh "$TAR"
