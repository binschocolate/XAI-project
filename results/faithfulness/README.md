# 단계 6: Faithfulness 평가 결과 (1차)

- 날짜: 2026-11-19 · 실행: ariel 서버 (RTX A5000), 팀원 checkpoint 12개(`final.pt`) 그대로 사용
- 대상: ResNet-18/50 × ρ 0.95/0.5 × seed 0–2, `splits/test_xai800.csv` 800장 (그룹당 200장)
- Attribution (단계 5): Grad-CAM(layer4), Integrated Gradients(픽셀 black baseline, 50 step), RISE(마스크 4000, 7×7, p=0.5, black fill), Random(대조군).
  설명 대상 class = 모델 예측 class

## 설정
| 항목 | 값 |
|---|---|
| 제거 단위 | 14×14 grid patch (16px, 196개). patch 중요도 = patch 안 attribution 평균 |
| 곡선 | 4 patch 마다 1점 (50점), x = 지운 비율 [0,1], 사다리꼴 AUC |
| 점수 | target class softmax 확률 |
| MIF (= Deletion) | 중요한 patch 부터 지움. 낮을수록 좋음 |
| LIF | 덜 중요한 patch 부터 지움. 높을수록 좋음 |
| Insertion | "중요 k개만 남은 상태" = LIF 상태와 같아서 `insertion_auc = lif_auc` |
| SRG | LIF − MIF (Blücher et al. 2024). 클수록 좋음 |
| method=random | 무작위 순서 곡선 (R-OMS 용 대조군) |
| `p_full_baseline` | 전부 지운(baseline 만 남은) 이미지의 target 확률 |

Baseline (정규화 전 픽셀 공간에서 채움)
| baseline | 내용 |
|---|---|
| black | 픽셀 0 |
| mean | train+val pool 채널 평균 [0.418, 0.430, 0.357] |
| blur | 원본 Gaussian blur (kernel 41, sigma 8) |
| inpaint | 남은 픽셀로부터 multi-scale 확산(push-pull) inpainting (딥러닝 아님) |
| trainset | 이미지마다 고정된 무작위 train+val 이미지의 같은 위치 픽셀 (seed 0) |

## 파일
- `<run>/auc.csv`: 이미지 × method × baseline 한 줄 (16,000줄). `aligned` = 1 이면 배경-클래스 일치 그룹
- `<run>/summary.csv`: subset(all / aligned / conflicting) × method × baseline 평균
- `summary_all.csv`: 12개 run 의 summary.csv 를 합친 것
- 곡선 원본(`curves.npz`)과 attribution map(`.npy`)은 용량 때문에 git 제외 (ariel `outputs/` 에 있음)
- `figs/`: `fig1_srg_by_baseline.png`(baseline별 SRG), `fig2_curves_resnet18/50.png`(MIF·LIF 곡선), `fig3_trainset_drop.png`(Black→Train-set SRG 변화), `fig4_group_srg.png`(aligned/conflicting별 SRG) — `python src/plot_faithfulness.py`로 생성

## 재현
```bash
python src/attribution.py   --data_dir $DATA_DIR --runs <run> --vis          # 단계 5 -> outputs/attributions/<run>/
python src/faithfulness.py  --data_dir $DATA_DIR --runs <run>                # 단계 6 -> outputs/faithfulness/<run>/
python src/plot_faithfulness.py                                              # 그림
python src/check_ckpts.py   --data_dir $DATA_DIR                             # (선택) checkpoint 로드·팀 결과 대조
```
checkpoint 는 `models/<run>/final.pt`, 데이터는 `metadata.csv` 가 있는 Waterbirds 폴더.

## 1차 관찰 (seed 평균, 통계 검정 전)
- Random 순서 SRG ≈ 0 (−0.002 ~ 0.007) → 측정 정상
- SRG: RISE·Grad-CAM 0.21–0.42, IG 0.02–0.12 (IG 는 LIF 가 Random 과 비슷)
- 기법 순위는 대부분 RISE > Grad-CAM > IG. 순위 변화는 주로 RISE↔Grad-CAM 교체이며 **train-set baseline** 에서 일관되게 Grad-CAM 이 1위
- 팀 정의 민감도(1 − 평균 Kendall τ, 기법 3개)는 shortcut / clean 사이에 일관된 차이 없음 (기법 3개라 τ 가 너무 거침)
- 대신 Black → Train-set 에서 SRG 하락 폭이 shortcut 이 더 큼: Grad-CAM R18 −0.12 vs clean −0.05, R50 −0.11 vs −0.03 / RISE R18 −0.19 vs −0.14, R50 −0.14 vs −0.09
  (가설: train-set 은 지운 자리에 다른 이미지의 배경을 넣어서, 배경에 의존하는 shortcut 모델이 더 흔들림 → 그룹별 분석·통계 검정 필요)
