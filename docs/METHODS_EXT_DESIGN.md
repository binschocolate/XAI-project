# 기법 확장 견고성 분석: SmoothGrad · LIME 추가 (2026-11-24, 실행 전에 고정)

## 왜 하나
- 단계 7–9 의 순위 민감도 S 는 기법 3개(Grad-CAM·IG·RISE)의 순위로 계산해 매우 거칠다 (가능한 τ 값이 몇 개뿐).
  실제로 주 분석 D 는 seed 마다 +0.40 / 0.00 / −0.27 처럼 부호가 바뀌었다.
- 기법을 5개로 늘리면 순위 비교 쌍이 3 → 10 개가 되어 S 가 촘촘해진다.
  "H2 효과가 없다"와 "지표가 거칠어 안 보였다"를 구분하려는 목적.

## 주 분석과의 관계 (중요)
- **기법 3개 결과(`results/analysis/`)가 주 결과다. 이 분석은 그것을 대체하지 않는다.**
- 이 분석은 사후에 추가한 **견고성 분석**으로 보고한다. 결과와 무관하게 아래 설정과 지표를 그대로 보고한다.

## 설정
| 기법 | 설정 |
|---|---|
| SmoothGrad (Smilkov et al. 2017) | 노이즈 샘플 50개, σ = 0.15 × (입력 최대 − 최소), target logit 의 입력 gradient 평균 → 절댓값 → 채널 합 |
| LIME (Ribeiro et al. 2016, `lime_image` 기본값 따름) | SLIC superpixel (n_segments=50, compactness=10), 샘플 1,000개 (첫 샘플 = 원본), 꺼진 superpixel 은 그 superpixel 평균색, cosine 거리 kernel (width 0.25) 가중 Ridge(α=1), target softmax 확률 |

- 나머지는 단계 5·6 과 동일: 원래 모델 12개, test_xai800, target = 원래 예측 class, 14×14 patch, baseline 5종, 50점 곡선
- 기존 Grad-CAM·IG·RISE·Random map 은 재사용 (다시 만들지 않음). faithfulness 는 6개 기법을 함께 새로 평가 (`outputs/faithfulness_m6/`)

## 보고할 것
1. 기법별 SRG·MIF (단계 6 과 같은 표)
2. 기법 5개 순위 민감도 S, D = S_shortcut − S_clean (scope all / aligned / conflicting), seed 별 값과 이미지 bootstrap 구간
3. 확인용: 같은 faithfulness 결과에서 기법 3개만으로 다시 계산한 S 가 기존 주 결과와 같은지 (fp16 배치 구성 차이로 미세한 차이 가능)

## 해석 기준 (미리 고정)
| 결과 | 해석 |
|---|---|
| 기법 5개 D 가 두 arch·seed 다수에서 양수 | 주 분석의 미지지는 지표 해상도 때문일 수 있음 (단, 주 결론은 바꾸지 않고 "확장 분석에서는 …"으로 보고) |
| 기법 5개 D 도 방향이 섞임 | 지표를 촘촘히 해도 shortcut 의 순위 불안정성이 일관되게 크지 않음 → 주 분석의 미지지를 뒷받침 |
