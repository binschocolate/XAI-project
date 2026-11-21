# 단계 7–9 분석 결과 (1차)

- 날짜: 2026-11-21 · 실행: ariel 서버 (CPU), `python src/analysis_stage7_9.py` (약 18초)
- 입력: 단계 5·6 결과 (`results/faithfulness/` 의 auc.csv 와 같은 데이터 + ariel 로컬의 curves.npz, attribution meta.csv)
- 명세: `docs/PROJECT_PLAN.md` "단계 7–10 상세 실행 계획" (2026-11-21)
- **단계 5·6 구현은 ariel 실행본 기준이다.** PROJECT_PLAN 9-3에 적힌 다른 로컬 구현과 baseline 설정이 다르므로, 공식 구현이 바뀌면 이 분석은 다시 돌린다 (입력만 바꾸면 됨).

## 공통 정의
- 순위 대상: Grad-CAM, IG, RISE (Random 은 대조군, 순위 제외). Q = −MIF 또는 SRG (클수록 좋음)
- 순위 민감도 S = 1 − 평균 Kendall τ-b (baseline 5종의 10쌍). 정확한 동점은 공동 순위, τ 정의 불가 쌍은 제외하고 유효 쌍 수 보고. 구현은 scipy 와 인공 입력 300개로 대조 검증
- scope: all / aligned(그룹 0·3) / conflicting(그룹 1·2) / 4개 개별 그룹. scope 평균 = 그룹 평균 동일 가중
- bootstrap: 4그룹 각 200장 복원추출, 2,000회, seed 0. 같은 추출을 모든 run·method·baseline 에 적용하고 매 반복 평균 → 순위 → S → D 재계산.
  **구간은 고정된 모델에 조건부인 평가 이미지 불확실성이며 학습 seed 수준의 유의성이 아니다.**

## 파일
| 파일 | 단계 | 내용 |
|---|---|---|
| `validation_report.md` | 7-0 | 입력 검증 171항목 (모두 통과) |
| `faith_auc.csv` | 7-1 | run × method × baseline × scope 평균 MIF/LIF/SRG, 표본 수, MRG |
| `roms.csv` | 7-1, 9-2 | Random 순서 AUC, p_orig, p_full_baseline |
| `score_variation.csv` | 7-2 | baseline 5종에 대한 평균 점수 SD(ddof=0)·Range |
| `sensitivity.csv` | 7-2 | S, FlipRate, 유효 쌍 수, baseline별 순위 |
| `method_pair_differences.csv` | 7-3 | 방법 쌍 차이 d + paired bootstrap 95% 구간, baseline 간 부호 변화 |
| `paired_comparison.csv` | 8-1 | 같은 arch·seed 의 D = S_shortcut − S_clean |
| `bootstrap_summary.csv` | 8-3, 8-4, 9-1 | D, arch 평균 D, G, A, I 의 95% 구간 |
| `group_comparison.csv` | 9-1 | scope 별 D, I = D_conflicting − D_aligned |
| `robustness.csv` | 9-4 | Black 제외 / baseline 하나씩 제외 / 두 모델 모두 정답 이미지 |
| `figs/` | 8-1, 8-2, 9-2 | paired plot, SRI–S 산점도, Random 순서 곡선 |

## 결과 요약 (seed 평균)

### 단계 7
- 순위 민감도 S (scope=all): MIF — R18 clean 0.178 / shortcut 0.222, R50 0.178 / 0.089 · SRG — R18 0.267 / 0.267, R50 0.356 / 0.311
- 점수 변동 SD 의 shortcut/clean 배율: Grad-CAM·RISE 1.3–4.2배 (두 arch, MIF·SRG 모두 >1), IG 0.6–1.0배
- 방법 쌍: IG 는 항상 최하위라 IG 쌍은 거의 역전 없음. Grad-CAM–RISE 는 SRG 기준 12/12 run 에서 baseline 에 따라 부호가 바뀜 (Train-set 에서 Grad-CAM 우위), MIF 기준 5/12 run

### 단계 8 (주 분석)
| | 평균 D | seed별 D | D>0 seed | arch 평균 D 95% 구간 |
|---|---|---|---|---|
| R18 MIF (주 결과) | +0.044 | +0.40, 0.00, −0.27 | 1/3 | [−0.04, 0.27] |
| R50 MIF (주 결과) | −0.089 | −0.27, 0.00, 0.00 | 0/3 | [−0.22, 0.00] |
| R18 SRG | 0.000 | 0, 0, 0 | 0/3 | [−0.09, 0.09] |
| R50 SRG | −0.044 | −0.13, 0, 0 | 0/3 | [−0.13, 0.13] |

- **H2 미지지** (방향 불일치, 구간이 0 포함)
- SRI–S 산점도: 단조 경향 없음 (탐색)
- SRG 안정화: G = S_MIF − S_SRG 가 12개 모델 모두 음수 (−0.04 ~ −0.22) → SRG 가 더 불안정. A 는 seed별 방향 혼재

### 단계 9 (탐색)
- 9-1 aligned 그룹 MIF D: R18 +0.67 [0.36, 0.76], R50 +0.71 [0.49, 0.89] (두 arch 모두 seed 3/3 양수). conflicting: R18 +0.04, R50 −0.18. I 는 모든 arch·seed 에서 음수
  - 해석 후보: aligned 에서 shortcut 모델의 세 기법 MIF 차이(최대−최소)가 0.017–0.026 으로 매우 작음 (clean 0.063–0.074) → 근소한 차이로 순위 역전. 4개 개별 그룹의 주도 그룹은 arch 마다 다름
- 9-2 Random MIF AUC 는 baseline 에 따라 다름 (Black 0.64 최저, Blur·Inpaint 0.69–0.74). shortcut 은 conflicting 에서 더 빨리 떨어지지만 원본 확신도 자체가 낮음(0.89–0.91 vs 0.96–0.97) → 순수 OOD 효과로 단정 불가
- 9-3 baseline 구현 문서화: 미실행 (PROJECT_PLAN 9-3 을 실제 실행 설정으로 수정 필요)
- 9-4 견고성: Train-set 제외 시 R18 MIF D +0.26, Blur 제외 시 −0.11, 공통 정답 이미지 R50 MIF −0.44 → 주 결과의 부호가 baseline 구성·subset 에 따라 바뀜 (공통 정답은 점추정만)

## 남은 것
- 9-3 문서화, 공통 정답 subset bootstrap, validation_report 에 실행 commit hash·명령 기록 추가
- 단계 10 (보고서·발표)
