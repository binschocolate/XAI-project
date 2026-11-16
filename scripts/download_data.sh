#!/usr/bin/env bash
# Waterbirds tar.gz를 NAS에 한 번만 받아둔다 (압축은 여기서 풀지 않음).
#   bash scripts/download_data.sh
# GPU 노드에서 외부 접속(DNS)이 안 되면 master 노드에서 실행해도 된다 (다운로드만 함).
set -eo pipefail

TAR_DIR=/data/$USER/datasets
TAR=$TAR_DIR/waterbird_complete95_forest2water2.tar.gz
URL=https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz

mkdir -p "$TAR_DIR"
if [ -f "$TAR" ] && [ "$(stat -c %s "$TAR")" -gt 100000000 ]; then   # 100MB 이상이면 정상으로 간주
  echo "already exists: $TAR"
else
  rm -f "$TAR"
  # .part로 받은 뒤 성공했을 때만 이름을 바꾼다 (실패 시 빈 파일이 남지 않게)
  wget -c -O "$TAR.part" "$URL"
  mv "$TAR.part" "$TAR"
fi
ls -lh "$TAR"
