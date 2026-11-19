#!/usr/bin/env bash
# Waterbirds를 "내 컴퓨터(Mac/Linux)"에서 받아 Seraph용 tar로 만든다.
#   bash scripts/prepare_data_local.sh            # 현재 폴더에 tar 생성
# 이후 출력되는 scp 명령으로 Seraph NAS에 올린다. 자세한 설명: docs/DATASET.md
#
# 왜 로컬에서?
#  - 원 배포처(nlp.stanford.edu)는 접속 불가 → WILDS 배포본(CodaLab)을 사용
#  - WILDS 압축은 상위 폴더 없이 풀리므로, 스크립트들이 기대하는 폴더 이름으로 다시 묶어야 함
#  - Seraph master에서는 압축 해제가 금지라 이 작업을 로컬에서 함
set -eo pipefail

URL="https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/"
NAME=waterbird_complete95_forest2water2
WORK=${WORK:-./waterbirds_tmp}

mkdir -p "$WORK" && cd "$WORK"
[ -s archive.tar.gz ] || curl -L -o archive.tar.gz "$URL"
ls -lh archive.tar.gz

rm -rf "$NAME" && mkdir "$NAME"
tar -xzf archive.tar.gz -C "$NAME"

LINES=$(wc -l < "$NAME/metadata.csv" | tr -d ' ')
DIRS=$(find "$NAME" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
echo "metadata.csv lines: $LINES (expected 11789) / species folders: $DIRS (expected 200)"
[ "$LINES" = "11789" ] || { echo "ERROR: unexpected metadata size"; exit 1; }

# macOS의 ._ 숨김 파일이 들어가지 않게 COPYFILE_DISABLE=1
COPYFILE_DISABLE=1 tar -czf "$NAME.tar.gz" "$NAME"
ls -l "$NAME.tar.gz"

echo
echo "다음 명령으로 Seraph에 업로드 (~/.ssh/config의 Host 이름이 moana인 경우):"
echo "  scp $(pwd)/$NAME.tar.gz moana:/data/<USER>/datasets/"
echo "업로드 후 양쪽 크기(ls -l)가 같으면 $(pwd) 폴더는 지워도 된다."
