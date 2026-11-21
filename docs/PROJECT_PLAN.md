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
| **마스킹으로 생기는 OOD**와 **Waterbirds의 ID/OOD group(분포 이동)**을 구분 | 두 요인을 분리해서 분석 (aligned vs conflicting 그룹 비교 + random 순서 곡선으로 masking 반응을 진단; 정보 제거 효과와 순수 OOD를 동일시하지 않음) |
| Blücher et al. (2024) 참고 | R-OMS(random 순서 곡선), SRG(= LIF − MIF) 지표를 도입 |

## 3. 가설

- **H1 (재현)**: masking baseline이 바뀌면 같은 XAI 기법의 점수와 기법 간 순위가 바뀐다.
- **H2 (주 가설)**: shortcut 의존도(SRI)가 높은 모델일수록 baseline에 따른 순위 변화(민감도)가 크다. SRG 지표에서는 이 효과가 약하다.
- **H3 (탐색: group과 masking 반응)**: shortcut–clean의 민감도 차이가 aligned/conflicting 조건에서 어떻게 달라지는지, random masking 반응과 어떤 관계를 보이는지 조사한다. 이 분석만으로 masking OOD의 인과효과를 식별하거나 두 원인을 완전히 분리했다고 주장하지 않는다.

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
| 4 | **Shortcut reliance 검증**: test 평가 + SRI | `metrics.json`, `sri_summary.csv` → git: `results/<arch>/` |
| 5 | Attribution 생성: Grad-CAM, Integrated Gradients, RISE (+ Random 기준) | attribution 캐시 |
| 6 | Faithfulness 평가: MIF(Deletion)·LIF × baseline 5종; SRG = LIF AUC − MIF AUC | 이미지별 AUC·곡선·설정 파일 |
| 7 | 결과 검증 → AUC 집계 → 점수 변동·Kendall τ 기반 순위 민감도·방법 쌍 비교 | 점수표·민감도표·방법 쌍 차이 |
| 8 | 같은 arch·seed의 shortcut–clean paired 비교, 이미지 bootstrap, SRI 경향·SRG 안정화 분석 | H2 결과·불확실성 구간·paired plot |
| 9 | 조건·원인 탐색: 그룹별 차이, R-OMS/random 곡선, baseline 제외·공통 정답 분석 | H3 탐색 결과·견고성 분석 |
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


### 단계 7–10 상세 실행 계획 (2026-11-21)

> 아래는 실행할 분석의 명세이며, 단계 7–10의 실행 완료나 가설 지지를 의미하지 않는다. 진행 상태는 `PROGRESS.md`에서 관리한다.
> GitHub main의 설계와 로컬 `exp/faithfulness`의 구현(`ecd5158`, attribution `0e37386`)을 검토해 작성했다. 단계 5·6 담당자는 실제 실행 버전·설정을 결과와 함께 전달한다. 로컬 구현 내용을 main에 이미 반영한 것으로 간주하지 않는다.

#### 공통 원칙과 분석 우선순위

**주 질문:** 실제 배경 의존성이 높은 모델은 masking baseline 변경에 따른 XAI 평가 순위의 불안정성이 더 큰가? SRG는 이 불안정성을 줄이는가?

| 우선순위 | 분석 | 해석 |
|---|---|---|
| 주 분석 | 전체 800장에서 같은 arch·seed의 shortcut–clean **MIF 순위 민감도** 차이 | H2의 직접적인 비교 |
| 주요 보조 분석 | SRG 순위 민감도와 shortcut–clean 격차의 감소 | SRG 안정화 여부 |
| 보조 분석 | 방법별 점수 변동, SRI–민감도 산점도, 방법 쌍별 차이 | 변화의 크기·구조 확인 |
| 탐색 분석 | aligned/conflicting·4그룹, Random 곡선, baseline 제외·공통 정답 분석 | 결과가 나타나는 조건과 대안 설명 탐색 |

- 결과를 본 뒤 유리한 지표·baseline·이미지 subset으로 주 분석을 교체하지 않는다. 변경이 필요하면 이유와 시점을 기록하고 원래 결과도 남긴다.
- 모델 이름의 clean은 상대적으로 배경 의존성이 낮은 학습 조건을 뜻한다. 배경을 전혀 사용하지 않는다는 보장은 아니다.
- 학습 SRI는 클래스 내 **서로 다른 이미지 그룹의 평균 확률 차이**다. 동일 이미지의 배경만 교체한 counterfactual 효과로 표현하지 않는다.
- 800장은 4그룹 각 200장인 **group-balanced XAI subset**이다. 전체 평균은 원래 test의 클래스 빈도를 반영한 평균과 다르다.
- 그룹 비교는 aligned/conflicting 용어를 사용한다. 특히 ρ=0.5 모델에서 conflicting을 무조건 OOD라고 부르지 않는다.
- shortcut 모델이 배경을 이용한다면 배경을 지목하는 설명도 faithful할 수 있다. 새에 집중하는지 여부만으로 설명의 우열이나 결과 채택 여부를 결정하지 않는다.

#### 단계 7-0. 결과 수령 및 완전성 검증

팀원에게 run별 평균만 받지 않고 이미지별 결과와 재현 정보를 받는다.

| 입력 | 용도 |
|---|---|
| `auc.csv` | 이미지별 MIF·LIF, SRG 계산과 재표집 |
| `curves.npz` | 곡선 형태·끝점·AUC 검증 |
| 평가 `config.json` | baseline·grid·삭제 비율·점수·난수 설정 |
| attribution `meta.csv`, `config.json` | 이미지 순서·설명 대상·IG/RISE 설정 |
| 모델 `metrics.json`, `test_preds.csv` | SRI·정확도·정답 여부 연결 |
| `test_xai800.csv`, 실행 commit hash, 명령·로그 | 동일 subset과 실행 버전 확인 |

검증 순서:

1. 12개 run에 동일한 800장과 그룹별 200장이 있는지 확인한다. `idx`뿐 아니라 `img_filename`으로 run 간 대응을 검증한다.
2. attribution과 평가의 이미지 순서·예측 대상 클래스가 같은지 확인한다. 삭제 과정에서는 **원본의 예측 클래스**를 고정하며 매 단계 argmax로 바꾸지 않는다.
3. `(run, img_filename, method, baseline, curve)` 키의 누락·중복을 검사한다. Random 별도 실행 결과를 합칠 때 중복시키지 않는다.
4. Grad-CAM·IG·RISE·Random, baseline 5종, MIF·LIF가 모두 있으면 run당 `800 × 4 × 5 × 2 = 32,000`행, 12개 총 `384,000`행이다. 부분 실행은 본 결과와 구분한다.
5. 유한값 여부와 확률·MIF/LIF AUC의 범위(0–1), 파생 SRG 범위(-1–1)를 검사한다.
6. 삭제 비율 0의 시작점은 원본 예측 확률과 맞아야 하고, 같은 이미지·baseline의 완전 삭제 끝점은 설명 방법 간 일치해야 한다.
7. CSV AUC와 저장 곡선의 재적분 결과를 비교한다. 곡선 저장 dtype(float16)과 추론 정밀도를 고려한 허용 오차를 기록한다.
8. MIF/LIF가 단조롭지 않다는 이유만으로 실패 처리하지 않는다. 특히 부호 있는 IG에서 음의 증거를 제거하면 확률이 증가할 수 있다.
9. SRI 누락·NaN, 불완전한 method×baseline 표를 그대로 순위 계산에 넣지 않는다. `pivot_table`의 중복 평균이나 `groupby`의 결측 제외가 오류를 숨기지 않도록 먼저 검사한다.

**완료 기준:** 입력 검증 기록이 있고, 누락·중복·subset 불일치가 해결된 결과만 본 분석에 포함한다.

#### 단계 7-1. 이미지별 지표와 집계표 작성

- 이미지별로 `SRG = AUC_lif − AUC_mif`를 계산한다.
- `run × method × baseline × scope`별 평균 MIF·LIF·SRG와 표본 수를 저장한다.
- scope는 `all`, `aligned`, `conflicting`, 4개 개별 그룹으로 둔다.
- 기본 800장에서는 이미지 평균과 4그룹 동일 가중 평균이 같다. 정답 조건 등으로 subset을 만들면 그룹별 표본 수를 보고하고, 해당 scope 안 그룹 평균을 동일 가중한다. 필요한 그룹이 비면 그 비교는 계산 불가로 표시한다.
- Random은 대조군으로 별도 저장하고, XAI 3종의 순위에 포함하지 않는다.

**산출물:** `faith_auc.csv`(평균·표본 수·scope), `roms.csv`(Random AUC).

#### 단계 7-2. 점수 민감도와 순위 민감도 계산

**점수 민감도:** 모델 m, 방법 a, 지표 q(MIF 또는 SRG)에서 baseline b별 평균 점수의 표준편차와 최대–최소 차이를 계산한다.

```text
V(m,a,q) = SD_b[mean_image AUC(m,a,b,q)]
Range(m,a,q) = max_b mean_AUC − min_b mean_AUC
```

표준편차의 정의(ddof)를 고정한다. 여기서는 비교하는 5개 baseline 집합의 기술통계로 ddof=0을 사용한다. 서로 다른 척도의 MIF·SRG 원시 표준편차만으로 SRG가 우수하다고 판단하지 않는다.

**순위 민감도:** baseline마다 Grad-CAM·IG·RISE를 평균 점수로 정렬한다(MIF 낮을수록, SRG 높을수록 좋음). baseline 5종의 10쌍에 대해 Kendall τ를 구한다.

```text
S(m,q) = 1 − mean_{b<b'} tau(rank_b, rank_b')
FlipRate(m,q) = 전체 순위가 달라진 baseline 쌍 수 / 10
```

- S=0이면 모든 baseline에서 같은 순위다. 큰 값은 더 큰 불안정성을 뜻하며 백분율이 아니다.
- 점수를 반올림한 뒤 순위를 만들지 않는다.
- 현재 요약 코드의 동점 없는 순위 가정을 보완한다. 정확한 동점은 공동 순위로 두고 Kendall τ-b 등 동점 처리 방식을 사용한다. FlipRate도 임의의 정렬 순서 대신 공동 순위 관계를 비교한다.
- 모든 방법이 동점이라 τ가 정의되지 않는 경우는 비교 불가로 표시하고, 유효 쌍 수를 보고한다. 조용히 제외한 결과를 완전한 10쌍 결과와 동일하게 취급하지 않는다.
- 순위 변화와 점수 변화는 별개 결과로 보고한다. 근소한 차이의 불확실성은 단계 8의 bootstrap으로 확인한다.

**산출물:** `sensitivity.csv`, `score_variation.csv`, baseline별 순위표. 민감도표에 arch·rho·seed·SRI·scope·metric·유효 쌍 수를 포함한다.

#### 단계 7-3. 방법 쌍별 순위 역전 분석

세 쌍(Grad-CAM–IG, Grad-CAM–RISE, IG–RISE)을 비교한다. 좋은 방향을 통일해 `Q = −MIF AUC` 또는 `Q = SRG`로 정의한다.

```text
d(a,a',b) = mean_image[Q(a,b) − Q(a',b)]
```

- d>0이면 앞 방법이 더 좋다. baseline 변경에 따라 부호가 바뀌면 순서 역전이다.
- 같은 이미지의 두 방법 차이를 이용해 paired bootstrap 구간을 계산한다.
- 차이가 작고 구간이 0을 포함하면 “순위는 바뀌었으나 우열은 불명확”으로 해석한다.
- 전체 순위 불안정성이 어떤 방법 쌍에서 발생했는지 제시한다. 다수의 방법·baseline 비교는 탐색 결과로 보고하며, 각각을 독립적인 확증 검정처럼 해석하지 않는다.

**단계 7 완료 기준:** 입력 검증 기록, 평균 점수표, 점수 변동표, 순위 민감도표, 방법 쌍 차이표가 만들어진다. 이 단계만으로 H2 지지를 선언하지 않는다.

#### 단계 8-1. 같은 아키텍처·seed의 shortcut–clean paired 비교

아키텍처별로 seed 0–2를 대응시킨다. 같은 seed는 train split 구성과 초기화의 대응 단위이지 두 조건의 학습 이미지가 완전히 같다는 뜻은 아니다.

```text
D(arch,seed,q,scope) =
    S(arch,rho=.95,seed,q,scope) − S(arch,rho=.50,seed,q,scope)
```

- D>0: shortcut 모델이 더 불안정하다.
- D≈0: 차이가 작다.
- D<0: 예상과 반대 방향이다.

주 결과는 `scope=all, q=MIF`로 둔다. R18·R50을 각각 보고하고, seed별 D, 평균 D, seed 간 표준편차, D가 양수인 seed 수를 제시한다. clean–shortcut 점을 seed별로 연결한 paired plot을 만든다.

| arch | metric | clean 평균 S | shortcut 평균 S | 평균 paired 차이 D | D>0인 seed 수 |
|---|---|---|---|---|---|
| R18 | MIF | 결과 | 결과 | 결과 | /3 |
| R18 | SRG | 결과 | 결과 | 결과 | /3 |
| R50 | MIF | 결과 | 결과 | 결과 | /3 |
| R50 | SRG | 결과 | 결과 | 결과 | /3 |

#### 단계 8-2. SRI–민감도 경향 확인 (보조)

- 모델 12개의 SRI–S 산점도를 MIF·SRG별로 만든다. arch는 색, rho는 점 모양으로 구분하고 seed별 점을 모두 표시한다.
- 주 근거는 arch 내부 shortcut–clean 비교다. 전체 12점의 상관계수 하나로 인과관계나 연속적인 dose-response를 주장하지 않는다.
- rho는 2수준뿐이며, 관찰된 SRI 4수준에는 arch 차이도 섞인다. 회귀를 한다면 arch를 포함한 소규모 탐색 분석으로 제한하고 불안정성을 명시한다.
- SRI와 정확도·초기 confidence가 함께 달라질 수 있으므로, 단순 상관만으로 shortcut 의존성이 유일한 원인이라고 결론 내리지 않는다.

#### 단계 8-3. 이미지 불확실성과 학습 반복 변동 구분

**그룹별 paired 이미지 bootstrap:**

1. 기본 분석에서 4그룹 각각 200장을 복원 추출한다(반복 2,000회, 분석 난수 seed 고정).
2. 동일한 이미지 추출 목록을 모든 방법·baseline·대응 모델에 적용한다. 공통 800장을 쓰는 run 전반에서 이미지 대응을 유지한다.
3. 매 반복마다 평균 AUC → 순위 → S → D를 다시 계산한다. 기존 순위를 고정하거나 이미지별 순위를 평균하지 않는다.
4. arch별 평균 D도 동일 반복의 seed별 D를 평균해 계산한다.
5. 2.5%, 97.5% 분위수 구간을 제시하고, 순위 지표가 이산적이라 구간이 계단형 또는 퇴화할 수 있음을 기록한다.
6. 공통 정답 subset은 고정된 적격 이미지 집합 안에서 그룹별 원래 표본 수만큼 재표집한다. subset 크기와 그룹별 수를 보고한다.

**해석:**
- 이 구간은 **고정된 학습 모델에 조건부인 평가 이미지 표본의 불확실성**이다. 새 학습 반복까지 포괄하는 구간이 아니다.
- 학습 반복은 조건별 seed 3개다. 이미지 800장을 독립적인 모델 실험 800회로 간주하지 않는다.
- 두 arch가 같은 seed의 학습 split을 공유하므로 6개 paired 차이를 무조건 독립 표본으로 풀링하지 않는다.
- seed별 효과와 방향·두 arch의 재현 여부를 중심으로 보고한다. 작은 p-value를 얻기 위해 이미지·baseline 쌍·방법을 독립 반복처럼 사용하지 않는다.
- seed 기반 검정을 추가하면 작은 표본 수와 가정을 명시하고 탐색적으로 보고한다. 이미지 bootstrap 구간이 0을 제외한다는 사실을 학습 반복 수준의 유의성으로 바꾸어 표현하지 않는다.

**산출물:** `paired_comparison.csv`, `bootstrap_summary.csv`, `method_pair_differences.csv`. 구간의 재표집 단위·가중 방식·난수 seed·반복 수를 함께 저장한다.

#### 단계 8-4. SRG 안정화 여부 확인

```text
G(model,scope) = S_MIF − S_SRG
A(arch,seed,scope) = D_MIF − D_SRG
```

- G>0: 해당 모델의 순위가 SRG에서 더 안정적이다.
- A>0: SRG에서 shortcut–clean 민감도 격차가 감소한다.
- 두 질문을 구분해 seed별 결과를 보고하고, 같은 paired bootstrap 표본으로 구간을 계산한다.
- SRG가 반드시 안정화한다고 가정하지 않는다. 효과가 없거나 반대인 결과도 보고한다.
- **순위 안정성은 평가의 정확성·faithfulness의 정답을 증명하지 않는다.** “baseline 변경에 더 안정적이었다”를 우선적인 결론으로 사용한다.

**단계 8 완료 기준:** paired 비교표·seed별 plot·SRI 산점도·이미지 bootstrap 구간·SRG 비교가 완성되고 H2에 대해 지지/부분 지지/미지지를 결과에 맞게 기술한다.

#### 단계 9-1. 그룹 조건에 따른 차이 탐색

단계 7·8을 aligned/conflicting에서 반복하고 다음 차이를 계산한다.

```text
I(arch,seed,q) = D_conflicting − D_aligned
```

- I>0이면 shortcut–clean 차이가 conflicting에서 더 크다.
- 4개 개별 그룹도 확인해 특정 클래스가 결과를 주도하는지 점검한다.
- subgroup 분석은 주 분석을 대체하지 않는다. 두 그룹을 나눴다는 사실만으로 masking 효과와 분포 이동의 인과효과가 완전히 분리됐다고 주장하지 않는다.

#### 단계 9-2. R-OMS/Random 곡선과 초기 confidence 점검

- 동일한 무작위 patch 순서로 삭제했을 때 원본 예측 클래스 확률 곡선과 AUC를 baseline별로 비교한다.
- 같은 모델 내 baseline 비교를 우선하고, clean/shortcut·aligned/conflicting 차이와 원본 확률 `p_orig`도 함께 제시한다.
- Random MIF와 LIF는 기대상 대칭이지만 유한 표본에서는 다를 수 있다. Random SRG는 기대상 0이며 모든 실행에서 정확히 0이어야 하는 검증 조건은 아니다.
- Random 곡선은 특정 XAI 순서와 무관한 masking 반응을 보여주지만, **실제 정보 제거·대체값·confidence 차이도 포함한다. R-OMS가 낮다는 사실을 순수한 OOD 크기로 단정하지 않는다.**
- Random AUC와 순위 불안정성의 관계는 탐색적으로 해석한다. 공통 출력에 기반한 관계이므로 독립적인 원인 증명으로 사용하지 않는다.

Random 보정의 해석도 구분한다.

```text
MRG(method) = AUC_random − AUC_MIF(method)
```

동일 모델·scope·baseline에서 모든 방법에 같은 Random AUC를 적용하면 MIF의 순위는 바뀌지 않는다. 따라서 MRG만 계산하고 “OOD 보정 후 순위가 안정화됐다”고 설명하지 않는다. SRG 분석은 별도로 수행한다.

#### 단계 9-3. baseline의 실제 구현과 대안 설명 기록

로컬 구현을 검토한 조건이며, 실제 실행 config와 일치하는지 단계 5·6 담당자와 확인한다.

| baseline | 구현 조건 | 해석 시 주의 |
|---|---|---|
| Black | 정규화 전 픽셀 0 | 정규화 공간의 0과 다르다 |
| Mean | ImageNet 채널 평균색 | Waterbirds 실측 평균색으로 기술하지 않는다 |
| Blur | 원본 이미지 Gaussian blur, σ=10px | 저주파 정보가 남으므로 완전한 정보 부재가 아니다 |
| Inpainting | normalized convolution, σ=16px | Telea·diffusion이 아닌 근사 방식; 전체 삭제 시 fallback도 명시 |
| Train | 공식 train에서 고정 seed로 뽑은 100장 donor pool, 평가 이미지마다 그중 **한 장**의 같은 위치 사용 | 100장 예측 평균이 아니다. donor가 새 클래스·배경 증거를 추가할 수 있다 |

- IG는 검정 기준·부호 유지 채널 합, RISE는 검정 fill을 사용한다. 설명 생성 baseline과 평가 baseline은 별도 개념이다.
- Grad-CAM은 layer4의 낮은 해상도에서 보간되고 IG·RISE와 점수 성질이 다르다. 모두 14×14 patch로 모았다는 이유만으로 생성 방식까지 동일해지는 것은 아니다.
- 14×14 grid는 196개 16px patch이고, 삭제 비율의 약 50개 지점에서 AUC를 계산한다. “50개 patch를 삭제”로 표현하지 않는다.
- Insertion은 현재 계획의 **동일 keep-mask·동일 대체 연산을 쓰는 상보적 경로**에서 LIF와 역방향으로 대응하므로 별도 실행하지 않는 것으로 정리한다. 모든 Insertion 구현과 모든 inpainting 정의에 무조건 같은 AUC라고 일반화하지 않는다.

#### 단계 9-4. 견고성 분석과 추가 실험 조건

기존 결과를 재집계하는 아래 분석을 우선한다.

| 분석 | 확인할 대안 설명 |
|---|---|
| Black 제외 | IG·RISE의 내부 Black baseline과 평가 Black의 일치가 결과를 주도하는가 |
| baseline 하나씩 제외(5개 모두 보고) | 특정 대체 방식 하나가 결론을 좌우하는가 |
| 같은 arch·seed의 두 모델이 모두 정답인 공통 이미지 | 오분류·서로 다른 설명 대상 클래스가 차이를 주도하는가 |

- baseline 제외 시 쌍 수도 10쌍에서 6쌍으로 변한다. 각 조건에서 S를 다시 계산하고 같은 조건의 clean–shortcut을 비교한다. 유리한 제외 조합만 선택하지 않는다.
- 공통 정답 분석은 선택된 표본에 대한 보조 분석이며 전체 800장 결과를 대체하지 않는다. 네 그룹 수와 가중을 기록한다.
- 초기 confidence·예측 클래스 구성도 함께 확인한다. 공통 정답 조건만으로 confidence 차이까지 제거됐다고 주장하지 않는다.
- Random 순서, Grad-CAM 동점 patch 순서, RISE 마스크, Train donor는 고정 난수 조건이다. 이미지 bootstrap이 이 난수 변동까지 평가하는 것은 아니다.
- 결과가 근소한 역전이나 특정 donor/난수 조건에 강하게 의존할 때만 고정 subset에서 난수 반복·RISE 수렴·donor 반복 등을 추가 확인한다. 보고서에는 확인 범위를 명시한다.
- ROAR, 새 rho 수준, pixel 단위 비교, diffusion inpainting은 이번 주 분석의 필수 조건으로 늘리지 않는다. 추가 실행은 관찰된 문제를 해결할 필요가 있을 때 범위를 정해 수행한다.

**단계 9 완료 기준:** 그룹별 D·I, Random 곡선, baseline 구현 설명, 세 가지 재집계 견고성 결과와 남은 대안 설명을 정리한다.

#### 단계 10. 시각화·발표·보고서 통합

| 그림·표 | 전달할 내용 |
|---|---|
| 실험 설계표 | 12개 모델·800장·3방법·5baseline·MIF/LIF/SRG |
| SRI·worst-group accuracy | 모델의 배경 의존성이 실제로 다름(단계 4 결과 재사용) |
| baseline별 평균 AUC·순위 heatmap | 점수·순위가 어떻게 달라지는지 |
| shortcut–clean paired plot | H2의 직접적인 결과, seed별 점 표시 |
| SRI–민감도 산점도 | arch·rho를 구분한 보조 경향 |
| MIF–SRG 비교 | 전체 안정화 G와 격차 감소 A |
| 그룹별 결과·Random 곡선 | 변화가 두드러지는 조건 |
| 방법 쌍 차이·견고성 표 | 근소한 역전과 특정 baseline 의존성 |

발표는 연구 질문 → 모델 검증 → baseline별 변화 → 주 비교 → SRG → 조건·원인 탐색 → 한계 순서로 구성한다. 본문에는 핵심 paired 결과를 두고 세부 방법 쌍·제외 분석은 부록으로 보낼 수 있다.

보고서에서 반드시 구분할 사항:

- 학습 rho는 2수준, seed는 조건별 3개이고 arch와 SRI 차이가 섞여 있다.
- SRI는 개별 이미지의 배경 교체 효과가 아니다.
- 800장은 group-balanced subset이며 이미지 구간과 학습 반복 불확실성은 다르다.
- Random/R-OMS는 독립적인 OOD 검출 결과가 아니다.
- baseline·설명 생성 방식·고정 난수의 한계를 명시한다.
- 순위가 안정적이라는 사실과 평가가 정확하다는 주장은 다르다.
- 그룹별·Random 분석만으로 두 원인의 인과효과를 완전히 분리했다고 주장하지 않는다.

결론은 관찰에 맞춰 작성한다.

| 결과 | 가능한 결론 |
|---|---|
| 두 arch·seed에서 같은 방향 | 검증된 배경 의존성이 높은 모델에서 더 큰 baseline 순위 민감도가 관찰됨 |
| 일부 조건에서만 차이 | 관계가 arch·group·평가 지표·baseline 구성에 따라 달라짐 |
| 차이가 작거나 반대 | 배경 의존성 차이가 있어도 순위 불안정성이 반드시 증가하지는 않음 |
| SRG에서 S 감소 | SRG가 이번 조건에서 baseline 변경에 더 안정적이었음; 평가 정확성의 증명은 아님 |

**단계 10 완료 기준:** 주 결과표·그림·분석 설정·한계가 정리되고 실행 버전과 산출물 위치를 추적할 수 있다. `PROGRESS.md`의 단계 상태는 실제 실행과 검증이 끝난 뒤 업데이트한다.

#### 권장 분석 산출물과 구현 순서

아래 파일명은 추가 구현할 산출물 명세이며 이미 생성된 결과가 아니다.

| 순서 | 작업 | 산출물 |
|---|---|---|
| 1 | 입력 완전성·설정 검사 | `validation_report.md` |
| 2 | 기존 `src/summarize_faith.py` 보완: 누락·동점 처리, 4그룹·표본 수, 점수 변동·방법 쌍 | `faith_auc.csv`, `sensitivity.csv`, `score_variation.csv`, `roms.csv` |
| 3 | paired 비교·bootstrap·SRG 비교 | `paired_comparison.csv`, `bootstrap_summary.csv`, `method_pair_differences.csv` |
| 4 | 그룹·baseline 제외·공통 정답 분석 | `group_comparison.csv`, `robustness.csv` |
| 5 | 시각화·보고서 통합 | 그림·표, 단계 7–10 분석 기록, 발표·보고서 |

- 원본 이미지별 결과는 재분석 가능하도록 보존한다. 작은 요약표·분석 설정·그림을 git으로 공유하고 큰 캐시·checkpoint는 기존 저장 정책을 따른다.
- 분석 중 코드의 결과 유실·중복 평균·순위 방향·SRG 계산·이미지 pairing을 작은 인공 입력으로 검증한다. 문서 추가만으로 분석 코드 검증이 끝난 것으로 간주하지 않는다.
- 팀원의 단계 5·6 실행과 병렬로 입력 명세·분석 코드를 준비하고, 원본 결과가 도착하면 위 순서로 실행한다.

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
