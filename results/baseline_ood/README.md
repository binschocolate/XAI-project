# Masking baseline 별 OOD 정도 정량화 (탐색 분석)

- 날짜: 2026-11-23 · 실행: ariel 서버 (RTX A5000) · 코드: `src/baseline_ood.py`, 요약·그림: `src/plot_baseline_ood.py`
- 배경: proposal Abstract 의 "baseline별 OOD 정도를 정량화" + 교수님 피드백 3 (마스킹 OOD 와 group 분포 이동 구분).
  단계 7–9 결과를 본 뒤 실행했으므로 **탐색 분석**으로 보고한다.

## 방법
**같은 patch 를 같은 비율로 지우고 채우는 값(baseline)만 바꾼다** → 지운 정보량이 같으므로 baseline 간 차이 = 채운 값의 효과.

- 마스크: `splits/test_xai800.csv` 800장, 14×14 grid(196 patch), 이미지마다 고정 seed(1000 + 이미지 번호)의 무작위 patch 순서에서 앞 25 / 50 / 75% 제거
- baseline 5종: 단계 6과 동일 구현 (Black 픽셀 0, Mean = train+val 평균색, Blur k41 σ8, Inpaint push-pull, Train-set = 단계 6과 같은 donor)
- 특징: 마지막 avgpool 출력 (R18 512차원, R50 2048차원)
- ① **OOD 정도**: 각 모델의 학습 분포 특징 대비 거리
  - Mahalanobis (Lee et al. 2018): 클래스별 평균 + 공통 공분산(Ledoit-Wolf), 클래스 중 최솟값 — **주 지표**
  - KNN (Sun et al. 2022): L2 정규화 특징, 10번째 최근접 학습 이미지까지 거리 (방향만 보므로 보조)
  - **AUROC(원본 test 이미지 vs 마스킹 이미지)**: 0.5 = 원본과 구분 불가, 1.0 = 완전히 구분(OOD)
- ② **정보 잔존**: 원본 특징과의 cosine 유사도, 원본 예측과의 일치율, 원본 예측 class 확률(`p_target`)
- 모델 13개: 우리 12개(학습 분포 = 각자의 B안 split 1,678장) + **ImageNet ResNet-50**(Waterbirds 미학습, 학습 분포 = train+val 5,994장, 클래스 없는 단일 Gaussian) — 모델과 무관한 "이미지 자체의 부자연스러움" 기준
- 원본(제거 0%) 행 = non-masking 기준점 (AUROC 0.5, cosine 1, 일치 1)

## 파일
- `ood_all.csv`: 13개 run × baseline × 비율 × scope(all / aligned / conflicting)
- `summary_fraction025.csv`: 25% 제거, scope=all 의 모델 유형별 평균
- `runs/<run>/{ood.csv, config.json, scores.npz}`: run 별 원본 결과 (`scores.npz` = 이미지별 점수)
- `figs/fig_ood_vs_retention.png`: 25% 제거, 정보 잔존 vs OOD AUROC
- `figs/fig_ood_by_fraction.png`: 제거 비율별 OOD AUROC
- `figs/fig_ptarget_by_group.png`: 25% 제거, 그룹별 원래 예측 class 확률

## 결과 (seed 평균)

### 1. baseline 별 OOD 순서는 모델과 무관하게 일정
25% 제거, Mahalanobis AUROC:

| | Black | Mean | Train-set | Blur | Inpaint |
|---|---|---|---|---|---|
| ImageNet R50 (ref) | 1.00 | 1.00 | 0.98 | 0.91 | 0.90 |
| R18 clean / shortcut | 1.00 / 0.99 | 0.96 / 0.96 | 0.90 / 0.90 | 0.81 / 0.83 | 0.80 / 0.81 |
| R50 clean / shortcut | 1.00 / 1.00 | 0.99 / 0.99 | 0.97 / 0.96 | 0.87 / 0.83 | 0.86 / 0.82 |

- 5종류 모델 모두 **Black > Mean > Train-set > Blur ≈ Inpaint** (ImageNet 기준 모델도 동일 → 이미지 자체의 성질)
- 정보 잔존(cosine)은 거의 반대: Blur·Inpaint 0.81–0.86 > Mean·Train-set 0.71–0.77 > Black 0.60–0.63
- 50% 이상 제거하면 모든 baseline 이 AUROC 0.94 이상 → baseline 비교는 25%가 가장 유익. 75%에서 Train-set AUROC 가 내려가는 것은 이미지 대부분이 다른 실제 사진이 되기 때문

### 2. 가설과의 관계
- **H3-1 (Black·Blur 같은 단순 마스킹이 OOD 생성):** Black 은 가장 OOD (예상대로), **Blur 는 가장 덜 OOD** (예상과 반대) → 부분 지지
- **OOD 정도 ≠ 순위 불안정의 원인:** 단계 7 에서 순위 역전을 주도한 것은 Train-set (Blur·Inpaint 와 SRG 순위 12/12 역전) 이고,
  가장 OOD 인 Black 은 Mean·Inpaint 와 거의 역전 없음 (0–3/12) → 평가 흔들림은 "OOD 정도"보다 "채운 자리에 들어간 내용"과 관련
- **H3-2 (shortcut 이 OOD 에 더 민감):** OOD 정도 자체는 shortcut / clean 차이 ≤ 0.05 (Mahalanobis), KNN 으로는 shortcut 이 Blur·Inpaint 를 오히려 덜 OOD 로 봄.
  aligned / conflicting 사이 OOD 차이도 0.01–0.02 → OOD 측면의 민감도 차이는 작음

### 3. 그룹별 예측 유지 (그림 `fig_ptarget_by_group.png`)
25% 제거 시 원래 예측 class 확률:

| | 원본 | Black | Blur | Inpaint |
|---|---|---|---|---|
| R18 shortcut, aligned | 0.98 | 0.78 | 0.90 | 0.90 |
| R18 shortcut, conflicting | 0.91 | 0.58 | 0.60 | 0.61 |
| R18 clean, conflicting | 0.96 | 0.65 | 0.84 | 0.84 |
| R50 shortcut, aligned | 0.99 | 0.76 | 0.95 | 0.95 |
| R50 shortcut, conflicting | 0.89 | 0.60 | 0.69 | 0.70 |

- clean 은 정보를 많이 남기는 Blur·Inpaint 에서 예측이 잘 유지됨
- **shortcut 은 conflicting 그룹에서만 Blur·Inpaint 의 이점이 거의 사라짐** — 그런데 OOD 정도는 그룹 간 같음
- 해석 후보: Blur·Inpaint 는 배경 색·질감을 남기고, shortcut 모델은 그것을 근거로 씀 → 배경이 정답과 어긋난 이미지에서 반대 class 로 끌림.
  즉 그룹별 차이는 **마스킹 OOD 가 아니라 남은 정보(배경 단서)와 분포 이동의 상호작용**으로 보임 (피드백 3 의 두 요인 분리)
- 주의: shortcut 모델은 conflicting 의 원본 확신도 자체가 낮음 (0.89–0.91 vs clean 0.96–0.97)

## 한계
- 단계 7–9 결과를 본 뒤 추가한 탐색 분석
- AUROC 는 원본 test 이미지 대비 상대값. ImageNet 기준 모델의 예측 일치율·`p_target` 은 1000-class 기준이라 해석하지 않음
- Mahalanobis 는 학습 1,678장으로 공분산을 추정 (Ledoit-Wolf shrinkage), KNN 은 방향만 비교
- 무작위 마스크 seed 고정 1회 (마스크 반복은 미실시)
