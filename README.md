# XAI-project (Team 6)

설명및신뢰가능한AI 텀프로젝트: **Shortcut 의존도에 따른 masking baseline별 XAI faithfulness 평가의 민감성** (Waterbirds, ResNet-18/50)

## 문서
- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md): 연구 질문, 피드백 반영, 실험 설계, 일정
- [`docs/PROGRESS.md`](docs/PROGRESS.md): **진행 현황 (작업할 때마다 업데이트)**
- [`docs/ANALYSIS_SRI.md`](docs/ANALYSIS_SRI.md): 단계 4 분석 (shortcut reliance 검증 결과 해석, 한계, 다음 단계 시사점)
- `docs/SERAPH_GUIDE.md`: Seraph(moana) 실행 명령 모음. **개인 정보가 들어갈 수 있어 git 제외 (각자 로컬에 보관)**

## 구조
```
src/
  common.py        시드, 전처리, 데이터셋, 모델
  make_splits.py   B안 train split (rho 0.95 / 0.5, seed 0-2) + test / XAI subset
  train.py         ERM 학습, 고정 epoch
  eval_sri.py      test 평가 + SRI (shortcut reliance index)
  aggregate.py     run별 결과를 표로 정리
scripts/
  setup_env.sh     conda env 'xai' 생성
  download_data.sh Waterbirds tar를 NAS에 다운로드
  stage_data.sh    tar를 GPU 노드 /local_datasets에 풀기
  train.sbatch     학습 + SRI 평가 (array 6개)
  export_results.sh Seraph 결과 중 작은 파일을 results/<arch>/로 복사
splits/            생성된 split CSV (git으로 공유)
results/           실험 결과 요약·로그 (git으로 공유, checkpoint 제외) → results/README.md
logs/              Slurm 로그 (git 제외)
```

## Seraph 경로
- 코드: `/data/$USER/repos/XAI-project`
- 데이터 tar: `/data/$USER/datasets/` → 실행 시 GPU 노드 `/local_datasets/$USER/`에 풀어서 사용
- 가중치 캐시: `/data/$USER/cache/torch`
- 결과: `/data/$USER/results/XAI-project/`
- `models/<arch>_rho<095|050>_seed<k>/`: `config.json`, `train_log.csv`, `final.pt`, `test_preds.csv`, `metrics.json`
- `sri_runs.csv`, `sri_summary.csv`: 전체 요약
- git 공유본: `bash scripts/export_results.sh <arch>` → `results/<arch>/`
