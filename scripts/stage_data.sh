#!/usr/bin/env bash
# NAS의 tar를 GPU 노드 로컬 디스크(/local_datasets)에 풀고 DATA_DIR을 export한다.
# 같은 노드에서 여러 job이 동시에 시작해도 한 번만 풀리도록 flock 사용.
# 사용:  source scripts/stage_data.sh   (GPU 노드에서)

TAR=/data/$USER/datasets/waterbird_complete95_forest2water2.tar.gz
LOCAL_ROOT=/local_datasets/$USER
export DATA_DIR=$LOCAL_ROOT/waterbird_complete95_forest2water2

mkdir -p "$LOCAL_ROOT"
(
  flock -x 200
  if [ ! -f "$DATA_DIR/.complete" ]; then
    echo "[stage] extracting $TAR -> $LOCAL_ROOT ($(hostname))"
    tar -xzf "$TAR" -C "$LOCAL_ROOT"
    touch "$DATA_DIR/.complete"
  fi
) 200>"$LOCAL_ROOT/.waterbirds.lock"
echo "[stage] DATA_DIR=$DATA_DIR"
