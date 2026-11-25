# ρ 수준 추가 실험 — 배경 의존도에 따른 경향 (dose-response)

- 날짜: 2026-11-25 · 실행: ariel (job 501842 smoke, 501850 본 실행, 새 모델 12개) · 설계(실행 전 고정): [`docs/RHO_LEVELS_DESIGN.md`](../../docs/RHO_LEVELS_DESIGN.md)
- 목적: 교수님 피드백 1 — "shortcut 에 대한 모델의 의존성이 높을수록 평가 결과가 더 민감하게 변하는가". 기존 ρ 2수준(0.5 / 0.95)에 0.75·0.99 를 더해 **4수준에서 경향**을 본다.
- **추가 실험이다. 기존 주 분석(`results/analysis/`, ρ 0.5 vs 0.95)은 그대로.**
- 코드: `src/make_splits_rho.py`(팀 `make_pair` 로 새 split, 기존 split 완전 재현 확인), `src/train.py`(`--rho` 에 0.75·0.99 허용만 추가), `src/dose_response.py`(분석)
- 새 모델 학습·평가 설정은 기존과 동일 (ImageNet 사전학습, SGD lr 1e-3, batch 32, 30 epoch, 마지막 checkpoint / attribution 6종·faithfulness 기존 설정 그대로)

## 파일
- `analysis/` — `summary.md`(요약), `trend.csv`(경향: Spearman·기울기·bootstrap 구간), `per_run.csv`(모델 24개 × 지표별 S·점수 변동), `validation_report.md`(338항목 통과)
- `models/<run>/` — 새 모델 12개의 config, train_log, metrics.json(SRI·worst-group), test_preds (checkpoint 는 ariel `models/` 에만)
- `faithfulness/<run>/`, `attribution_config/<run>.json` — run 별 faithfulness 요약·attribution 설정
- `figs/fig_sri_vs_S.png` — SRI vs 순위 민감도 (선 1개 = seed 1개의 ρ 4수준), `figs/vis_resnet18_rho099_seed0.png` — 예시 map
- 기존 12개 모델은 `results/methods_ext/` 와 같은 결과(`outputs/faithfulness_m6/`)를 합쳐 24개로 분석

## 결과 (seed 평균)
### 배경 의존도 — ρ 를 올리면 SRI 가 단계적으로 오른다
| SRI | ρ 0.5 | 0.75 | 0.95 | 0.99 |
|---|---|---|---|---|
| ResNet-18 | 0.014 | 0.121 | 0.409 | 0.607 |
| ResNet-50 | 0.005 | 0.074 | 0.344 | 0.573 |

새 모델 worst-group acc: R18 ρ0.75 0.66 / ρ0.99 0.19, R50 ρ0.75 0.79 / ρ0.99 0.36

### 점수 변동(baseline 5종에 대한 점수 SD) — 가장 뚜렷한 경향
- **Grad-CAM, RISE, LIME, SmoothGrad: 두 arch × MIF·LIF·SRG 모두 SRI 가 높을수록 커짐 (seed 3/3, Spearman 평균 0.47–1.0)**
  - 예: Grad-CAM R50 LIF 0.010 → 0.016 → 0.040 → 0.048 (ρ 0.5 → 0.99)
- IG 만 예외 (LIF·SRG 에서 반대 방향 또는 약함)

### 순위 민감도 S = 1 − 평균 Kendall τ (baseline 10쌍)
| S vs SRI | ResNet-18 | ResNet-50 |
|---|---|---|
| **주: 기법 3개, MIF, 전체 800장** | 양수 3/3, 기울기 +0.52 [0.28, 0.82] · S 0.18 / 0.09 / 0.22 / 0.49 | 양수 1/3, −0.18 [−0.51, 0.25] · S 0.18 / 0.18 / 0.09 / 0.09 |
| MIF, aligned (기법 3개) | 3/3, +1.19 [0.50, 1.32] · S 0.18 → 0.84 | 3/3, +1.36 [0.98, 1.70] · S 0.22 → 1.11 |
| MIF, aligned (기법 5개) | 3/3, +0.71 [0.37, 0.81] | 3/3, +0.54 [0.29, 0.77] |
| LIF, 전체 (기법 3개 / 5개) | 0/3, −0.42 [−0.59, −0.19] / 0/3, −0.53 [−0.66, −0.40] | 2/3, +0.13 [0.01, 0.36] / 1/3, −0.29 [−0.34, −0.07] |
| SRG, conflicting (기법 3개) | 0/3, −0.30 [−0.47, −0.23] | 0/3, −0.65 [−0.65, −0.44] |

(기울기 = S ~ SRI + seed 고정효과, [ ] = 그룹별 이미지 bootstrap 95%, 2,000회. ρ 0.99 의 R18 주 분석 상승이 경향 대부분을 만든다)

## 판정 (사전 기준)
- 주 분석: R18 만 증가, R50 은 반대 → 두 arch 일치하지 않음 → **순위의 단조 경향은 확인되지 않음 (기존 H2 미지지와 일관)**
- 보조: 점수 변동과 aligned MIF 순위 민감도는 두 arch 모두 의존도와 함께 단계적으로 증가 → 기준표의 **"보조 지표(점수 변동, aligned)에서만 경향 → 순위가 아니라 점수 수준·특정 조건의 효과로 보고"** 에 해당
- 한 줄: **shortcut 의존도가 높을수록 baseline 에 따른 faithfulness 점수 변화는 단계적으로 커지지만, 그것이 기법 순위 변화로 이어지는지는 지표(MIF/LIF/SRG)와 이미지 조건에 따라 다르다.**

## 한계
- arch 안에서만 경향을 봄 (arch 당 12점). ρ 0.99 의 waterbird-land 학습 이미지는 4장뿐
- 순위 지표는 기법 3개(쌍 3개)라 값이 거칠다 → 기법 5개로도 확인
