# ρ 수준 추가 실험: ρ = 0.75, 0.99 모델 (2026-11-25, 실행 전에 고정)

## 왜 하나
교수님 피드백 1: "shortcut 에 대한 모델의 의존성이 높을수록 평가 결과가 더 민감하게 변하는가"에 집중.
기존에는 ρ 2수준(0.5 / 0.95)이라 두 집단 비교만 가능했다. ρ 0.75·0.99 를 추가해 **4수준에서 경향(dose-response)** 을 본다.
기존 주 분석(ρ 0.5 vs 0.95, `results/analysis/`)은 그대로 두고, 이 실험은 **추가 실험**으로 보고한다.

## 모델 (새로 12개)
- ResNet-18·50 × ρ {0.75, 0.99} × seed 0–2. 학습은 팀 `src/train.py` 그대로 (ImageNet 사전학습, SGD lr 1e-3, batch 32, 30 epoch, 마지막 checkpoint)
- split: 팀 `make_pair(pool, seed, minority_frac = 1 − ρ)` 를 그대로 사용 (`src/make_splits_rho.py`).
  같은 함수로 기존 ρ 0.95·0.5 split 이 seed 3개 모두 **완전히 재현됨**을 확인. 모든 ρ 에서 1,678장, landbird 1,300 / waterbird 378 동일

| ρ | landbird-land | landbird-water | waterbird-land | waterbird-water |
|---|---|---|---|---|
| 0.5 (기존) | 650 | 650 | 189 | 189 |
| 0.75 | 975 | 325 | 94 | 284 |
| 0.95 (기존) | 1,235 | 65 | 19 | 359 |
| 0.99 | 1,287 | 13 | 4 | 374 |

## 평가 (기존과 동일)
- SRI·worst-group (팀 `eval_sri.py`, test 5,794장)
- attribution 6종 (Grad-CAM·IG·RISE·Random·SmoothGrad·LIME, 기존과 같은 설정), faithfulness 6종 (test_xai800, baseline 5종, 14×14 patch)
- 기존 12개 모델 결과(`outputs/faithfulness_m6/`)와 합쳐 **24개 모델**로 분석 (`src/dose_response.py`)

## 분석 (미리 고정)
- **주:** arch × seed 마다 SRI(ρ 4수준의 4개 모델)와 순위 민감도 S(기법 3개, MIF, 전체 800장)의 Spearman 상관 → 6개 중 양수 개수·평균.
  arch 별 기울기 (S ~ SRI + seed 고정효과) 와 이미지 bootstrap 95% 구간
- **보조:** 같은 방식으로 기법 5개 S, aligned·conflicting, LIF·SRG, 기법별 점수 변동 SD (LIF 는 2026-11-25 실행 전에 추가)

## 해석 기준
| 결과 | 해석 |
|---|---|
| 주 분석에서 두 arch 모두 seed 다수 양수 + 기울기 구간이 0 제외 | 의존도가 높을수록 평가가 더 민감 (경향 확인) |
| 방향이 섞이거나 기울기 구간이 0 포함 | 의존도 증가에 따른 단조 경향은 없음 → 기존 H2 미지지와 일관 |
| 보조 지표(점수 변동, aligned)에서만 경향 | 순위가 아니라 점수 수준/특정 조건에서만 나타나는 효과로 보고 |

## 한계 (미리 명시)
- arch 와 SRI 가 섞여 있으므로 arch 안에서만 경향을 본다. ρ 4수준 × seed 3 = arch 당 12점
- ρ 0.95·0.99 는 소수 그룹이 매우 적어(19·4장) 둘 사이 SRI 차이가 작을 수 있음. ρ 0.99 의 waterbird-land 는 4장뿐 (외워서 맞히는 수준)
