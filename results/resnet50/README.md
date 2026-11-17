# ResNet-50: shortcut reliance 검증 (단계 4)

- 날짜: 2026-11-17 · Slurm job: `220467` (pilot은 ResNet-18 결과를 그대로 적용)
- 설정: ResNet-18과 동일. ImageNet 사전학습 ResNet-50 (`IMAGENET1K_V1`), ERM, SGD (lr 1e-3, momentum 0.9, wd 1e-4), batch 32, augmentation 없음, **30 epoch 고정**, 마지막 checkpoint
- Train: B안 (ρ별 1,678장, seed 0–2) · Test: 공식 test 5,794장

## 요약 (seed 0–2, 평균 ± 표준편차)

| ρ | 모델 | avg acc | worst-group acc | group gap | SRI |
|---|---|---|---|---|---|
| 0.5 | clean | 0.953 ± 0.001 | 0.852 ± 0.006 | 0.005 ± 0.004 | 0.005 ± 0.002 |
| 0.95 | shortcut | 0.830 ± 0.006 | 0.541 ± 0.016 | 0.337 ± 0.004 | 0.342 ± 0.007 |

## run별

| run | avg acc | worst-group acc | group gap | SRI |
|---|---|---|---|---|
| resnet50_rho050_seed0 | 0.9534 | 0.8458 | -0.0000 | 0.0071 |
| resnet50_rho050_seed1 | 0.9539 | 0.8536 | 0.0059 | 0.0035 |
| resnet50_rho050_seed2 | 0.9520 | 0.8567 | 0.0077 | 0.0042 |
| resnet50_rho095_seed0 | 0.8245 | 0.5576 | 0.3414 | 0.3477 |
| resnet50_rho095_seed1 | 0.8360 | 0.5249 | 0.3369 | 0.3449 |
| resnet50_rho095_seed2 | 0.8298 | 0.5405 | 0.3324 | 0.3343 |

지표 정의는 [`../resnet18/README.md`](../resnet18/README.md#지표-정의)와 같다.

## 해석
- **검증 통과.** ρ = 0.95 모델은 seed 3개 모두에서 SRI 0.33–0.35, worst-group acc 0.52–0.56. clean은 SRI 0.004–0.007, worst-group 0.85–0.86.
- **ResNet-18과 비교:** 같은 ρ = 0.95로 학습해도 ResNet-50의 배경 의존이 더 약하다 (SRI 0.342 vs 0.410, worst-group 0.541 vs 0.339). clean 모델도 ResNet-50이 더 깨끗하다 (SRI 0.005 vs 0.014). 더 큰 사전학습 모델이 새 자체의 특징을 더 잘 활용하는 것으로 보인다.
- 그 결과 12개 모델의 SRI가 **네 수준**(R50 clean 0.005 < R18 clean 0.014 < R50 shortcut 0.342 < R18 shortcut 0.410)으로 나뉜다. 단계 7–8에서 "SRI가 클수록 baseline 민감도가 큰가"를 두 그룹 비교뿐 아니라 경향으로도 볼 수 있다.

## 파일
`runs/`, `sri_runs.csv`는 Seraph에서 `bash scripts/export_results.sh resnet50`로 채운다.
