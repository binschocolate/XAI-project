# H3-1 · H3-2 검증 실험 설계 (2026-11-23, 결과 보기 전에 고정)

> proposal 원문
> - **H3-1.** Black이나 Blur 같은 단순 마스킹 방식은 데이터에 OOD를 발생시킬 것이고, 이를 이용해 **모델의 판단에 사용되지 않는 부분**을 masking하면 XAI 신뢰성과 무관한 결과가 나올 것이다.
> - **H3-2.** H3-1에서의 결과는 shortcut model이 clean model보다 민감하게 반응할 것이다.
>
> 이전 탐색 분석(`src/h3_1_analysis.py`)은 "안 쓰는 부분"을 attribution(LIF)으로 정했다. 평가 대상인 설명으로 실험 조건을 정하는
> 순환 논리이고, RISE 는 자체적으로 black 마스킹을 써서 검사 대상(OOD 마스킹)이 조건에 섞인다. 이 문서의 실험은 **XAI 와 OOD 마스킹을 쓰지 않고**
> "안 쓰는 부분"을 정한다.

## 0. 공통
- 모델: 기존 12개 (ResNet-18·50 × clean ρ0.5 / shortcut ρ0.95 × seed 0–2), 평가 이미지: `splits/test_xai800.csv` 800장
- 새/배경 영역: **CUB-200-2011 segmentation** (`segmentations/<class>/<image>.png`, Waterbirds 파일명과 동일). 값 ≥ 128 = 새.
  모델 입력과 같은 전처리(Resize 256×256 → CenterCrop 224)를 nearest 로 적용
- baseline 5종: 단계 6과 동일 구현 (black, mean, blur, inpaint, trainset)
- 점수: 원래 모델이 예측한 class 의 softmax 확률 `p`. Δp = p(원본) − p(개입 후)
- **자연 배경 은행**: train+val pool 에서 장소(물/땅)별로 새가 작은 이미지 64장씩(새 면적 하위 순, seed 0 고정) 골라 새 영역을 push-pull inpaint 로 지운
  "새 없는 배경" 128장. 자연 교체는 모두 이 은행에서 가져온다 (분포 안의 실제 사진 픽셀, test 이미지는 쓰지 않음)
- OOD 정도: **ImageNet ResNet-50 (Waterbirds 미학습)** 특징의 Mahalanobis 거리, 원본 test 대비 AUROC. 입력 자체의 부자연스러움 (모델 반응과 분리)

## 1. H3-1 — clean 모델의 배경 (clean 6개만)
**"안 쓰는 부분" = clean 모델의 배경.** 근거: 배경이 정답과 50:50 무작위인 데이터로 학습 + SRI 0.004–0.014 (데이터 설계 + 행동), XAI 미사용.

### 실험 A: 마스킹 인공물
같은 배경 영역에 세 가지 개입, 이미지별 Δp:
1. baseline 5종으로 배경을 채움 (검사 대상)
2. **같은 장소** 자연 배경으로 교체 (땅→다른 땅, 물→다른 물; donor 4장 평균) — OOD 없는 기준
3. **반대 장소** 자연 배경으로 교체 (donor 4장 평균) — "clean 은 배경을 안 쓴다" 전제 확인 (= 배경 교체 테스트, 교수님 피드백 2)
- **인공물 = Δp(1, baseline b) − Δp(2)**
- 함께 기록: 예측 뒤집힘 비율, 개입 이미지의 OOD AUROC(ImageNet 기준)

### 실험 B: 평가가 속는가
정답을 아는 설명 3개를 단계 6과 같은 절차(14×14 patch, 50점 곡선, baseline 5종)로 평가:
- `bird` map: 새 = 중요 (clean 에게 맞는 설명), `background` map: 배경 = 중요 (clean 에게 **틀린** 설명), `random` map
- 같은 값을 가진 patch 사이 순서는 고정 seed 의 아주 작은 난수로 정함
- **부당한 점수 = MIF(random) − MIF(background)** (>0 이면 틀린 설명이 Random 보다 좋게 평가됨), 그리고 bird vs background 순위

### 판정 (clean 6개, seed·arch 별 + 이미지 bootstrap 95% 구간)
| 결과 | 해석 |
|---|---|
| 개입 3의 Δp 가 큼 (평균 > 0.05) | 전제 붕괴 → 해석 보류 |
| 인공물 > 0 이고 OOD AUROC 순서와 비례 (Spearman ρ > 0, seed 다수) **그리고** OOD baseline 에서 부당한 점수 > 0 | **H3-1 지지** |
| 모든 baseline 에서 인공물 ≈ 0 (\|평균\| < 0.01) | **H3-1 기각** (안 쓰는 곳을 OOD 로 가려도 흔들리지 않음) |
| 인공물은 있으나 OOD 순서와 무관 (예: trainset 최대) | 원인은 OOD 가 아니라 **채운 내용** |

## 2. H3-2 — 두 모델이 **공통으로 안 보는 칸** (같은 arch·seed 의 clean–shortcut 쌍 6개)
H3-2 를 "예측에 관련 없는 입력 영역을 가렸을 때, shortcut 모델이 마스킹으로 생긴 분포 변화에 더 민감한가"로 정의한다.
두 모델의 배경을 똑같이 가리는 비교는 **하지 않는다** (shortcut 은 배경을 쓰므로 정보 손실이 섞임).

### 단계 1: 안 보는 칸 찾기 (모델 행동, XAI·OOD 마스킹 미사용)
- 이미지를 **4×4 = 16칸** (56×56px) 으로 나눔
- 각 칸을 자연 배경 은행의 같은 위치 픽셀로 교체: **물 4장 + 땅 4장** (이미지마다 고정 seed)
- 8개 교체 모두에서 \|Δp\| < **τ = 0.02** 이면 그 모델이 "안 보는 칸". τ = 0.05 를 민감도 분석으로 함께 보고
- 같은 이미지에서 **clean 과 shortcut 모두** 안 보는 칸 = 공통 칸 C

### 단계 2: 공통 칸 마스킹
- C 전체를 baseline 5종으로 채움 → 두 모델 각각 Δp (같은 입력)
- 대조: C 를 같은 장소 자연 배경으로 교체 → Δp_ctrl (칸을 같이 바꿀 때의 상호작용 확인)
- 함께 기록: 개입 이미지의 OOD AUROC(ImageNet 기준), 두 모델의 특징 변화(원본 대비 cosine)
- **민감도 차이 = Δp_shortcut − Δp_clean** (같은 이미지·같은 입력)

### 진행 조건과 보고
- **진행 기준:** 이미지의 50% 이상에서 공통 칸이 2개 이상. 못 미치면 결과는 기록하되 주 결론으로 쓰지 않고,
  예비안(정답과 무관한 무늬를 학습 이미지에 심어 재학습 → 설계상 두 모델 모두 안 쓰는 영역)을 검토
- 반드시 보고: 이미지별 공통 칸 수 분포, 칸 위치 빈도(4×4), 새 면적과의 관계, τ 별 결과

### 판정 (쌍 6개 = 2 arch × 3 seed, 이미지 bootstrap 95% 구간)
| 결과 | 해석 |
|---|---|
| 진행 기준 미달 | 판정 보류 (공통으로 안 보는 영역이 부족) |
| 민감도 차이 > 0, 두 arch 모두 seed 다수, OOD 큰 baseline 에서 더 큼 | **H3-2 지지** |
| 민감도 차이 ≈ 0 | **H3-2 기각** (정보 손실이 없으면 shortcut 이 OOD 에 더 민감하지 않음) |
| 민감도 차이 < 0 | 반대 방향 |

## 3. 한계 (미리 명시)
- 칸 단위 "안 봄" 판정은 근사 (칸 하나씩 교체; 여러 칸을 같이 가릴 때의 상호작용은 단계 2 대조로만 확인)
- 자연 배경 은행의 inpaint 된 새 자리, trainset donor 의 새 포함 가능성
- 공통 칸이 가장자리에 몰리면 일반화가 제한됨
- 실험 A·B 는 clean 모델에만 적용 (shortcut 은 데이터 구조상 확실히 안 쓰는 영역이 없음)

## 4. 구현 · 실행
- `src/h3_prepare.py` — segmentation 정렬, 새/배경 mask, 자연 배경 은행, oracle map (CPU, 1회)
- `src/h3_experiment.py` — 실험 A, H3-2 단계 1·2 (GPU, 쌍 단위)
- 실험 B — 기존 `src/faithfulness.py` 에 oracle map 을 attribution 으로 넣어 실행
- `src/h3_analysis.py` — 판정표·그림 (CPU)
- `scripts/ariel/h3.sbatch` — array 6개 (쌍 = arch × seed), 동시 2개
