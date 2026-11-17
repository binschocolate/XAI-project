# 실험 결과 (git 보관용)

Seraph의 `/data/$USER/results/XAI-project/`에 저장된 결과 중 **작은 파일만** 여기로 옮겨 팀과 공유한다.
옮기는 방법: Seraph에서 `bash scripts/export_results.sh <arch>` → `git add results && git commit`

## 구조
```
results/
└── <arch>/                         # resnet18, resnet50
    ├── README.md                   # 요약 표 + 해석 (사람이 작성)
    ├── sri_runs.csv                # run별 지표 (aggregate.py 출력에서 해당 arch만)
    ├── runs/<arch>_rho<095|050>_seed<k>/
    │   ├── config.json             # 학습 설정
    │   ├── train_log.csv           # epoch별 train loss / acc / 그룹별 acc
    │   ├── metrics.json            # test: avg / 그룹별 / worst-group acc, group gap, SRI
    │   └── test_preds.csv          # test 5,794장 예측 (p_wb, pred)
    └── pilot/<run>/                # epoch 수 결정용 pilot (50 epoch, 평가 없음)
```

## git에 올리지 않는 것
- `final.pt` (checkpoint, R18 ~45MB / R50 ~95MB): Seraph `/data/$USER/results/XAI-project/models/`에만 보관
- `debug/` (1 epoch 테스트)

## 단계별 결과
| 단계 | 결과 | 상태 |
|---|---|---|
| 4. SRI 검증 | [`resnet18/`](resnet18/README.md) | ✅ |
| 4. SRI 검증 | [`resnet50/`](resnet50/README.md) | ✅ |
