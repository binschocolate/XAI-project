# ResNet-18: shortcut reliance 검증 (단계 4)

- 날짜: 2026-11-16 · Slurm job: pilot `220250`, 본 학습 `220260`
- 설정: ImageNet 사전학습 ResNet-18, ERM, SGD (lr 1e-3, momentum 0.9, wd 1e-4), batch 32, augmentation 없음, **30 epoch 고정**, 마지막 checkpoint
- Train: B안 (train+val pool에서 ρ별 1,678장, seed가 subsample과 초기화를 모두 결정)
- Test: 공식 test 5,794장 (클래스 안에서 배경 50:50)

## 요약 (seed 0–2, 평균 ± 표준편차)

| ρ | 모델 | avg acc | worst-group acc | group gap | SRI |
|---|---|---|---|---|---|
| 0.5 | clean | 0.936 ± 0.001 | 0.787 ± 0.020 | 0.016 ± 0.001 | 0.014 ± 0.001 |
| 0.95 | shortcut | 0.811 ± 0.005 | 0.339 ± 0.003 | 0.413 ± 0.009 | 0.410 ± 0.010 |

## run별

| run | avg acc | worst-group acc | group gap | SRI |
|---|---|---|---|---|
| resnet18_rho050_seed0 | 0.9356 | 0.7741 | 0.0160 | 0.0153 |
| resnet18_rho050_seed1 | 0.9377 | 0.7757 | 0.0172 | 0.0125 |
| resnet18_rho050_seed2 | 0.9351 | 0.8100 | 0.0153 | 0.0145 |
| resnet18_rho095_seed0 | 0.8114 | 0.3380 | 0.4086 | 0.4063 |
| resnet18_rho095_seed1 | 0.8155 | 0.3364 | 0.4068 | 0.4026 |
| resnet18_rho095_seed2 | 0.8048 | 0.3427 | 0.4241 | 0.4220 |

## 지표 정의
- **worst-group acc**: 4개 그룹(landbird/waterbird × land/water 배경) 중 가장 낮은 정확도
- **group gap**: 평균(aligned 그룹 acc) − 평균(conflicting 그룹 acc)
- **SRI**: 클래스별 \|평균 P(waterbird \| water 배경) − 평균 P(waterbird \| land 배경)\| 의 평균. test는 클래스 안에서 배경이 무작위 50:50이므로 배경의 효과로 해석

## 해석
- **검증 통과.** ρ = 0.95 모델은 seed 3개 모두에서 SRI 0.40–0.42, worst-group acc 0.34. clean은 SRI 0.012–0.015, worst-group 0.77–0.81. 네 지표가 모두 같은 방향이고 seed 간 편차가 작다.
- shortcut 모델은 같은 클래스에서 배경만 바뀌어도 P(waterbird)가 평균 약 41%p 움직인다. clean은 약 1.4%p.
- 두 모델은 train 크기·클래스 수가 같고 이미지 923장을 공유하므로, 차이는 배경 조합 비율에서 온 것으로 볼 수 있다.
- clean의 worst-group acc(0.79)가 평균(0.94)보다 낮지만 group gap이 0.016으로 작다 → 배경이 아니라 클래스 난이도/불균형(waterbird 378장 vs landbird 1,300장) 영향으로 보인다.

## Pilot (epoch 수 결정)
- ρ 0.95 / 0.5 seed 0, 50 epoch, test 평가 없음
- 두 모델 모두 ~10 epoch에 train acc 1.0, ~20 epoch에 loss ≈ 0.002로 수렴 → 여유를 두고 **30 epoch** 확정
- epoch당 2–3초 (RTX 3090)

## 파일
`runs/`, `pilot/`, `sri_runs.csv`는 Seraph에서 `bash scripts/export_results.sh resnet18`로 채운다.
