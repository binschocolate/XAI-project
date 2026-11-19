#!/usr/bin/env bash
# [더 이상 사용하지 않음]
# 원 배포처(nlp.stanford.edu)가 접속 불가이고, Seraph GPU 노드는 외부 DNS도 막혀 있어
# 이 방식으로는 받을 수 없다. 대신:
#   1) 내 컴퓨터에서  bash scripts/prepare_data_local.sh
#   2) 출력되는 scp 명령으로 /data/$USER/datasets/ 에 업로드
# 자세한 절차: docs/DATASET.md
echo "download_data.sh is deprecated. See docs/DATASET.md (use scripts/prepare_data_local.sh on your local machine)."
exit 1
