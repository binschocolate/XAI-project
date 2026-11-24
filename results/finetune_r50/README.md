# 마스킹 적응 추가 학습 실험 — ResNet-18 + ResNet-50 합본

- 날짜: 2026-11-24 · ResNet-50 실행: ariel job 499570 · ResNet-18 단독 결과는 `results/finetune/` (그대로 유지)
- 설계·설정은 ResNet-18 과 동일 (`results/finetune/README.md`): control(원본만) / maskadapt(원본 + 무작위 10–30% 를 baseline 5종으로 가림),
  5 epoch, SGD lr 1e-4, 두 조건 배치 순서 동일. 해석 기준도 동일 (사전 고정)
- 이 폴더의 표는 `src/compare_finetune.py` 를 R18+R50 으로 다시 실행한 것 (R18 수치는 `results/finetune/` 과 같음)

## 파일
- `summary.md`, `shortcut_check.csv`, `sensitivity_compare.csv`, `figs/fig_finetune_sensitivity_{resnet18,resnet50}.png`
- `<cond>/analysis/`, `<cond>/analysis_fixed/` — R18+R50 12개 run 의 단계 7–9 분석
- `models/<cond>/resnet50_*/` — R50 추가 학습 설정·로그·SRI, `faithfulness/<cond>/.../resnet50_*/` — R50 faithfulness 요약

## 결과 (seed 평균)
### Shortcut 의존성 (test 5,794장)
| | original | control | maskadapt |
|---|---|---|---|
| R50 shortcut SRI | 0.344 | 0.342 | 0.341 |
| R50 shortcut worst-group | 0.532 | 0.572 | 0.436 |
| R18 shortcut SRI | 0.409 | 0.412 | 0.377 |

→ R50 은 SRI 가 사실상 그대로 (**"SRI 비슷"**). 두 arch 모두 maskadapt 후 shortcut worst-group 하락

### 순위 민감도 S — shortcut 모델, MIF, aligned 그룹
| | original | control | maskadapt | 고정 순서 진단 (maskadapt) |
|---|---|---|---|---|
| R18 | 0.844 | 0.889 | 0.711 | 0.978 |
| **R50** | **0.933** | **0.933** | **0.578** | **0.756** |

- D(shortcut − clean, aligned MIF): R50 original +0.71 / control +0.67 / maskadapt +0.36 (세 조건 모두 seed 3/3 양수)
- SRG 순위 민감도는 조건 간 거의 동일

## 판정 (사전 기준)
- **R18:** SRI 비슷 + 역전 대부분 유지 → 마스킹 낯섦만으로 설명 어려움
- **R50:** SRI 비슷 + aligned 불안정성 약 40% 감소, 고정 순서 진단에서도 감소 → **마스킹 입력에 대한 미숙한 반응이 불안정성에 일부 기여**
- 종합: 낯섦은 일부 기여(R50)하지만 두 arch 모두 shortcut 의 불안정성이 남아 **그것만으로는 설명되지 않음**

## 한계
- 추가 학습 설정 1가지, 순위 지표가 거침 (전체 800장 수준의 작은 변화는 잡음 수준)
- maskadapt 후 worst-group 하락 → 판단 방식 변화가 일부 섞였을 수 있음
