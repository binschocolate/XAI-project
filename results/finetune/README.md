# 마스킹 적응 추가 학습 실험 (ResNet-18)

- 날짜: 2026-11-23 · 실행: ariel 서버 (RTX A5000)
- 질문: 기존 순위 불안정성이 **"모델이 마스킹 입력에 낯설어서"** 생긴 것인가?
- 코드: `src/finetune_masking.py`(추가 학습), 팀 `src/eval_sri.py`(SRI 재측정), `src/attribution.py`, `src/faithfulness.py`,
  `src/analysis_stage7_9.py`(옵션 `--n_runs --sri_root --fixed_order` 추가, 기본 동작은 이전과 동일함을 확인), `src/compare_finetune.py`(비교)

## 설계 (결과를 보기 전에 고정)
| 항목 | 설정 |
|---|---|
| 대상 | ResNet-18 clean/shortcut × seed 0–2 checkpoint 6개 |
| control | 원본 이미지만으로 추가 학습 (단순히 학습을 더 한 효과의 대조군) |
| maskadapt | 원본 + 무작위 patch 를 가린 이미지로 추가 학습 |
| 공통 | 각 모델의 B안 split 1,678장, 5 epoch, SGD lr 1e-4, momentum 0.9, wd 1e-4, batch 32, augmentation 없음, **두 조건 배치 순서 동일** |
| maskadapt 마스킹 | 이미지마다 확률 0.5 로 가림. 14×14 grid 중 10–30% 균등, 위치는 XAI 와 무관한 무작위, 채우는 값은 평가와 같은 baseline 5종 중 무작위 (trainset donor = 같은 배치의 다른 학습 이미지) |
| 재평가 | 같은 800장·baseline 5종·14×14 patch·50점 곡선. ① 새 attribution 으로 평가(**최종 결과**) ② 원래 모델의 attribution 순서·target 고정(**보조 진단**: 설명 변화 vs 입력 반응 변화 구분) |

해석 기준 (사전 고정)
- SRI 비슷 + maskadapt 에서 역전 감소 → 마스킹 입력에 대한 미숙한 반응이 불안정성에 기여했을 가능성
- SRI 비슷 + 역전 유지 → 마스킹에 대한 낯섦만으로 기존 역전을 설명하기 어려움
- SRI 크게 변함 → shortcut 의존성 변화가 섞여 원인 분리 어려움

## 결과 (seed 0–2 평균)

### 1. Shortcut 의존성 (test 5,794장)
| 조건 | clean SRI | shortcut SRI | shortcut worst-group | shortcut 정확도 |
|---|---|---|---|---|
| original | 0.014 | 0.409 | 0.332 | 0.812 |
| control | 0.012 | 0.412 | 0.368 | 0.803 |
| maskadapt | 0.004 | 0.377 | 0.261 | 0.834 |

- control 은 거의 변화 없음 (학습 loss 가 이미 ~0.001)
- maskadapt 는 마스킹 이미지 정확도 0.91–0.94 → 0.94–0.97 로 적응
- shortcut SRI 는 약 8% 감소했지만 clean 과의 차이는 유지 → **"SRI 비슷"** 으로 판정. 단 shortcut worst-group 이 0.33 → 0.26 으로 낮아져 판단 방식이 일부 바뀌었을 수 있음

### 2. 순위 민감도 S (shortcut 모델, 새 attribution = 최종 결과)
| | original | control | maskadapt |
|---|---|---|---|
| MIF, 전체 800장 | 0.222 | 0.400 | 0.089 |
| **MIF, aligned** | **0.844** | **0.889** | **0.711** |
| SRG, 전체 | 0.267 | 0.267 | 0.267 |

- aligned MIF 의 큰 불안정성은 대부분 유지: D(shortcut − clean) = original +0.67, control +0.71, maskadapt +0.53, **세 조건 모두 seed 3/3 양수**
- SRG 는 세 조건 모두 동일 (Train-set 에서의 Grad-CAM↔RISE 역전이 그대로)
- 전체 800장 MIF 의 변화(0.40 → 0.09)는 해석하지 않음: **거의 변하지 않은 control 에서도 S 가 0.22 → 0.40 으로 움직였음** → 기법 3개 순위 지표의 잡음 크기 안. bootstrap 구간도 겹침 (control D [−0.09, 0.22], maskadapt [−0.18, 0.09])

### 3. 보조 진단 (원래 attribution·target 고정)
- shortcut MIF aligned S: original 0.844, control 0.889, **maskadapt 0.978** → 같은 순서로 가린 입력에 대한 반응만 보면 오히려 더 불안정
- 따라서 최종 결과의 작은 감소(0.89 → 0.71)는 입력 반응 개선보다 **설명 자체의 변화** 쪽으로 보임

## 판정
**SRI 비슷 + 역전 대부분 유지 → 마스킹에 대한 낯섦만으로 기존 순위 역전을 설명하기 어려움.**
baseline OOD 정량화 결과(`results/baseline_ood/`: 순위 역전은 가장 OOD 인 Black 이 아니라 내용을 넣는 Train-set 과 배경 단서를 남기는 Blur·Inpaint 와 관련)와 같은 방향이다.

## 한계
- ResNet-18, 추가 학습 설정 1가지(5 epoch, lr 1e-4)
- 순위 민감도(기법 3개)가 거칠어 작은 변화는 잡음 수준 (control 변동이 근거)
- maskadapt 후 worst-group 하락 → 판단 방식 변화가 전혀 없다고 할 수 없음
- 이 실험은 "순위 변화의 몇 %가 OOD 때문"인지 분해하지 않음

## 파일
- `summary.md`, `shortcut_check.csv`, `sensitivity_compare.csv`, `figs/fig_finetune_sensitivity.png` — `src/compare_finetune.py`
- `<cond>/analysis/`, `<cond>/analysis_fixed/` — 조건·평가 방식별 단계 7–9 분석 결과 (CSV, 검증 보고서, 그림)
- `models/<cond>/<run>/` — 추가 학습 설정(`config.json`), 학습 로그, SRI·정확도(`metrics.json`). checkpoint 는 git 제외
- `faithfulness/<cond>/<faithfulness|faithfulness_fixed>/<run>/` — run 별 faithfulness 요약. 이미지별 auc.csv·곡선·attribution map 은 용량 때문에 ariel `outputs/ft/` 에만 있음
