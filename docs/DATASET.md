# 데이터셋 준비 (Waterbirds)

> 처음 한 번만 하면 된다. 팀원 각자 Seraph 계정에서 실험하려면 각자 자기 `/data/$USER/datasets/`에 올려야 한다.

## 요약
| 항목 | 내용 |
|---|---|
| 데이터 | Waterbirds (Sagawa et al., 2020). CUB-200 새 + Places 배경, 총 11,788장 |
| 받는 곳 | **WILDS 배포본 (CodaLab)**. 원 배포처 `nlp.stanford.edu`는 2026-11-16 기준 접속 불가 |
| 다운로드 주소 | `https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/` (WILDS 공식 코드 `wilds/datasets/waterbirds_dataset.py`의 `download_url`) |
| 크기 | 압축 약 467MB, 다시 묶은 tar 약 490MB |
| Seraph 위치 | `/data/$USER/datasets/waterbird_complete95_forest2water2.tar.gz` (NAS에는 tar 한 파일만) |
| 학습 시 위치 | GPU 노드 `/local_datasets/$USER/waterbird_complete95_forest2water2/` (`scripts/stage_data.sh`가 자동으로 풂) |

## 왜 이렇게 하나
- Seraph **GPU 노드**는 일부 외부 주소의 DNS가 막혀 있어 직접 다운로드가 안 된다.
- Seraph **master**는 압축 해제·python 실행이 금지라, 받은 압축을 검사하고 다시 묶는 작업을 할 수 없다.
- WILDS 압축은 상위 폴더 없이 풀린다. 우리 스크립트는 `waterbird_complete95_forest2water2/` 폴더 이름을 기대하므로 다시 묶어야 한다.
- 그래서 **내 컴퓨터에서 받고 → 검사·재압축 → Seraph로 업로드**한다.

## 절차

### 1. [내 컴퓨터] 받아서 다시 묶기
저장소 루트에서:
```bash
bash scripts/prepare_data_local.sh
```
스크립트가 하는 일: CodaLab에서 다운로드 → 압축 해제 → `metadata.csv`가 11,789줄(헤더 + 11,788장)인지 확인 → `waterbird_complete95_forest2water2.tar.gz`로 재압축.
작업 폴더는 기본 `./waterbirds_tmp` (`.gitignore`에 포함). 바꾸려면 `WORK=~/Downloads/wb bash scripts/prepare_data_local.sh`.

### 2. [내 컴퓨터] Seraph에 업로드
`~/.ssh/config`에 `Host moana`가 설정되어 있다면 (외부 접속 포트 30080 포함):
```bash
ssh moana "mkdir -p /data/<내 아이디>/datasets"
scp waterbirds_tmp/waterbird_complete95_forest2water2.tar.gz moana:/data/<내 아이디>/datasets/
```
VSCode Remote-SSH 탐색기에서 `datasets` 폴더로 드래그해도 된다.

### 3. 업로드 확인
```bash
ls -l waterbirds_tmp/waterbird_complete95_forest2water2.tar.gz
ssh moana "ls -l /data/<내 아이디>/datasets/waterbird_complete95_forest2water2.tar.gz"
```
두 바이트 수가 같으면 성공. 로컬 `waterbirds_tmp/`는 지워도 된다.

### 4. [Seraph GPU 노드] 데이터 검증
```bash
srun --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G -p debug_ce_ugrad --pty $SHELL
# 노드에서
source /data/$USER/anaconda3/etc/profile.d/conda.sh && conda activate xai
cd /data/$USER/repos/XAI-project
source scripts/stage_data.sh                      # /local_datasets/$USER 에 압축 해제
python src/make_splits.py --data_dir $DATA_DIR    # 원 논문 그룹별 개수와 다르면 에러로 멈춤
```
`make_splits.py`는 전체 11,788장, train+val pool `3965/650/189/1190`, test `2255/2255/642/642`를 검사한다. 통과하면 원 논문과 같은 split이다. (2026-11-16 확인 완료)

## 참고
- 보고서 표기: "Waterbirds (Sagawa et al., 2020), WILDS 배포본 사용"
- `metadata.csv` 컬럼: `img_id, img_filename, y (0=landbird, 1=waterbird), split (0=train, 1=val, 2=test), place (0=land, 1=water), place_filename`
- `scripts/download_data.sh`는 사용 중단 (Stanford 주소가 죽어 있음)
