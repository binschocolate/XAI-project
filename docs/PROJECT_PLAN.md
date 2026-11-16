# 프로젝트 계획서 (전체 목표와 방향)

> 설명및신뢰가능한AI (AI300800) 텀프로젝트 · Team 6 (강민서, 강지현, 박민기, 박현빈)
> 원본 proposal: "Masking Baseline에 따른 XAI Faithfulness 평가의 민감성: ID/OOD 환경과 Shortcut Learning을 중심으로"
> 이 문서는 큰 틀(무엇을, 왜)을 담는다. 진행 상황은 [`PROGRESS.md`](PROGRESS.md), Seraph 실행법은 `SERAPH_GUIDE.md`(로컬 전용).

---

## 1. 연구 질문

**모델이 shortcut(배경)에 많이 의존할수록, masking baseline을 바꿨을 때 XAI faithfulness 평가 결과(점수와 기법 간 순위)가 더 크게 흔들리는가?**

배경:
- Deletion/Insertion 같은 faithfulness 지표는 중요 영역을 지울 때 쓸 대체값(baseline)이 필요하다. Black, Mean, Blur, Inpainting 등 무엇을 쓰냐에 따라 같은 XAI 기법도 점수와 순위가 달라질 수 있다.
- Black, Blur 같은 단순 마스킹은 모델이 학습 중 본 적 없는 OOD 이미지를 만든다. 그러면 평가 점수가 faithfulness가 아니라 OOD 민감성을 반영할 위험이 있다.
- 배경 같은 spurious feature에 의존하는 shortcut 모델은 이런 OOD 입력에 더 민감할 것으로 예상한다.

## 2. 교수님 피드백과 반영 방향

| 피드백 | 반영 |
|---|---|
| "baseline에 따라 순위가 바뀐다"보다 **shortcut 의존도가 높을수록 평가가 더 민감하게 변하는가**에 집중 | 주 질문을 위처럼 바꿈. "순위가 바뀐다"(H1)는 재현 확인용으로만 사용 |
| shortcut / clean 모델의 차이를 **사전에 객관적으로 검증** (worst-group acc, 배경 변경에 따른 예측 변화 등) | 학습 후 **SRI(Shortcut Reliance Index)**, worst-group acc, group gap을 측정. 차이가 확인된 뒤에만 다음 단계로 진행 |
| **마스킹으로 생기는 OOD**와 **Waterbirds의 ID/OOD group(분포 이동)**을 구분 | 두 요인을 분리해서 분석 (aligned vs conflicting 그룹 비교 + random 순서 곡선으로 baseline 자체의 OOD 측정·보정) |
| Blücher et al. (2024) 참고 | R-OMS(random 순서 곡선), SRG(= LIF − MIF) 지표를 도입 |

## 3. 가설

- **H1 (재현)**: masking baseline이 바뀌면 같은 XAI 기법의 점수와 기법 간 순위가 바뀐다.
- **H2 (주 가설)**: shortcut 의존도(SRI)가 높은 모델일수록 baseline에 따른 순위 변화(민감도)가 크다. SRG 지표에서는 이 효과가 약하다.
- **H3 (OOD 분리)**: 마스킹 OOD 효과는 SRI가 높을수록 커지며, 데이터 분포 이동(group) 효과와는 별개로 나타난다.

## 4. 확정된 실험 설정

| 항목 | 설정 |
|---|---|
| 데이터 | 공식 Waterbirds (직접 합성하지 않음) |
| 모델 | ResNet-18, ResNet-50 (ImageNet 사전학습, fc만 2-class로 교체 후 전체 fine-tuning) |
| Shortcut 강도 | ρ = 0.95 (shortcut) / 0.5 (clean) |
| 반복 | seed 0, 1, 2. **seed가 train subsample과 학습 초기화를 모두 결정** |
| 모델 수 | 2 arch × 2 ρ × 3 seed = **12개** |
| 학습 | ERM, SGD lr 1e-3, momentum 0.9, weight decay 1e-4, batch 32, augmentation 없음 |
| 모델 선택 | **고정 epoch** (기본 30, pilot으로 확정), 마지막 checkpoint 사용. val로 고르지 않음 |
| Test | 공식 test 5,794장, 12개 모델 공통 |
| XAI 평가용 subset | test에서 그룹당 200장, 총 800장 (고정 seed로 1회 추출) |

### Train 구성 (B안)

공식 train(4,795) + val(1,199)을 합친 pool에서 양쪽 모두 1,678장을 추출한다. 50:50 데이터를 4,795장 만들 만큼 소수 조합 사진(물 위의 landbird, 땅 위의 waterbird)이 없기 때문에, 소수 그룹 수에 맞춰 크기를 줄였다.

| 그룹 | pool | Clean (ρ = 0.5) | Shortcut (ρ = 0.95) |
|---|---|---|---|
| landbird + land | 3,965 | 650 | 1,235 |
| landbird + water | 650 | 650 (전부) | 65 |
| waterbird + land | 189 | 189 (전부) | 19 |
| waterbird + water | 1,190 | 189 | 359 |
| **합계** | 5,994 | **1,678** | **1,678** |

- 두 모델의 **크기와 클래스 수(landbird 1,300 / waterbird 378)가 같고**, 배경 조합 비율만 다르다.
- Shortcut의 이미지는 Clean에 들어간 이미지를 최대한 재사용해서 뽑는다 (seed당 약 923장 공유).
- val을 학습에 쓰므로 별도 val은 없다. 고정 epoch으로 학습하므로 필요 없다. **test로 어떤 결정도 하지 않는다.**

## 5. 실험 파이프라인

| 단계 | 내용 | 산출물 |
|---|---|---|
| 1 | 환경 세팅, Waterbirds 다운로드 | conda env `xai`, tar on NAS |
| 2 | train split 생성 (B안, seed 3개) | `splits/*.csv` |
| 3 | 모델 학습 (pilot으로 epoch 확정 → ResNet-18 6개 → ResNet-50 6개) | `results/XAI-project/models/<run>/final.pt` |
| 4 | **Shortcut reliance 검증**: test 평가 + SRI | `metrics.json`, `sri_summary.csv` |
| 5 | Attribution 생성: Grad-CAM, Integrated Gradients, RISE (+ Random 기준) | attribution 캐시 |
| 6 | Faithfulness 평가: Deletion(MIF), Insertion, LIF, SRG × baseline 5종 (Black, Mean, Blur, Inpainting, Train-set) | AUC 결과 테이블 |
| 7 | 민감도 계산: baseline 쌍별 XAI 순위 Kendall τ → 민감도 = 1 − 평균 τ | 모델별 민감도 |
| 8 | 주 분석: shortcut vs clean 민감도 비교 + 통계 검정 | H2 결과 |
| 9 | 원인 분석: 그룹별(aligned/conflicting) 비교, random 순서 곡선(R-OMS)으로 baseline OOD 측정·보정 | H3 결과 |
| 10 | 그림 정리, 발표, 최종 보고서 | 발표 자료, 보고서 |

### 단계 4: SRI 정의

test set은 클래스 안에서 배경이 50:50 무작위로 배정되어 있으므로, 같은 클래스 안에서 배경에 따른 예측 확률 차이를 배경의 효과로 해석한다.

```
SRI = mean over y∈{landbird, waterbird} of | E[P(waterbird) | y, water 배경] − E[P(waterbird) | y, land 배경] |
```

함께 보고: 평균 정확도, worst-group accuracy, group gap = 평균(aligned 그룹 acc) − 평균(conflicting 그룹 acc).

**통과 기준**: ρ = 0.95 모델이 seed 3개 모두에서 SRI와 group gap은 확실히 높고 worst-group acc는 확실히 낮아야 한다. 아니면 학습 설정을 재검토한다.

### 구현 시 주의
- Black baseline은 **정규화 전 픽셀 공간**에서 0으로 대체한다 (정규화 후 0은 사실상 Mean baseline).
- proposal의 "Non-masking"은 곡선이 평평해져 baseline으로 의미가 없으므로 **Random 순서 대조군**으로 대체한다.
- IG의 baseline과 RISE의 fill 값도 일종의 baseline이다. 기본은 고정하고, 평가 baseline과 맞춘 조건은 여유가 있으면 확장 실험으로 한다.
- 제거 단위는 14×14 patch (pixel 단위 비교는 부록).

## 6. 시간이 부족할 때 빼는 순서
ROAR → pixel 단위 비교 → Diffusion inpainting → ResNet-50

## 7. 일정 (최종 발표: 15주차)

| 기간 | 목표 |
|---|---|
| ~11/22 | 환경·데이터·split, pilot, ResNet-18 학습, **SRI 검증** |
| ~11/29 | ResNet-50 학습, attribution 생성, baseline 구현 |
| ~12/6 | faithfulness 평가 전체, 민감도·주 분석, 원인 분석 |
| 발표 주 | 통계 정리, 그림, 발표 자료 → 이후 최종 보고서 |

## 8. 참고문헌
- Geirhos et al. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2020.
- Sagawa et al. Distributionally robust neural networks for group shifts. ICLR, 2020.
- Petsiuk et al. RISE. BMVC, 2018.
- Samek et al. Evaluating the visualization of what a deep neural network has learned. IEEE TNNLS, 2017.
- Hooker et al. A benchmark for interpretability methods in deep neural networks (ROAR). NeurIPS, 2019.
- Tomsett et al. Sanity checks for saliency metrics. AAAI, 2020.
- Sturmfels et al. Visualizing the impact of feature attribution baselines. Distill, 2020.
- Hase et al. The out-of-distribution problem in explainability and search for solutions. NeurIPS, 2021.
- **Blücher, Vielhaben, Strodthoff. Decoupling pixel flipping and occlusion strategy for consistent XAI benchmarks. arXiv:2401.06654, 2024.** (교수님 추천: R-OMS, SRG)
