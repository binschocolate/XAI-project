# 진행 현황 (항상 최신으로 유지)

> 작업할 때마다 이 파일을 업데이트한다. 새로 합류하거나 오랜만에 볼 때 **"지금 어디까지 했고 다음에 뭘 하나"**를 여기서 바로 알 수 있어야 한다.
> 큰 방향은 [`PROJECT_PLAN.md`](PROJECT_PLAN.md), Seraph 명령은 `SERAPH_GUIDE.md`(로컬 전용).

**마지막 업데이트:** 2026-11-18

---

## 지금 상태 한눈에

| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | 실험 설계 확정 (B안, 모델 12개) | ✅ 완료 |
| 1 | 저장소 / 코드 / Seraph 스크립트 준비 | ✅ 완료 |
| 1 | Seraph 환경 세팅 + 데이터 다운로드 | ✅ 완료 (11/16) |
| 2 | train split 생성 + git 커밋 | ✅ 완료 (11/16) |
| 3a | pilot 학습으로 epoch 수 확정 | ✅ 30 epoch (11/16) |
| 3b | ResNet-18 학습 (6개) | ✅ 완료 (11/16, job 220260) |
| 4 | SRI 검증 (ResNet-18) | ✅ 통과 (11/16) |
| 4 | ResNet-18 코드·결과 git 정리 (`exp/resnet18-sri` 브랜치, `results/resnet18/`) | ✅ (11/17) |
| 3c/4 | ResNet-50 학습 + SRI | ✅ 통과 (11/17, job 220467, `exp/resnet50-sri`) |
| 5 | Attribution 생성 | ⏳ **다음** |
| 6 | Faithfulness 평가 (baseline 5종) | ⬜ |
| 7–9 | 민감도, 주 분석, 원인 분석 | ⬜ |
| 10 | 발표 / 보고서 | ⬜ |

## 다음에 할 일 (위에서부터)
1. Seraph에서 run별 원본 파일 export: `exp/resnet18-sri`에 `export_results.sh resnet18`, `exp/resnet50-sri`에 `export_results.sh resnet50` → 두 브랜치 main에 merge
2. 단계 5 준비: attribution(Grad-CAM, IG, RISE) 생성 코드 작성 (`splits/test_xai800.csv` 사용)
3. 단계 6 준비: baseline 5종(Black, Mean, Blur, Inpainting, Train-set) + Deletion/Insertion/LIF/SRG 구현

## 결정 기록

| 날짜 | 결정 | 이유 |
|---|---|---|
| 2026-11-12 | 주 질문을 "SRI가 높을수록 baseline 민감도가 큰가"로 변경 | 교수님 피드백 1 |
| 2026-11-12 | shortcut/clean 차이를 SRI·worst-group acc로 사전 검증 | 교수님 피드백 2 |
| 2026-11-12 | 마스킹 OOD와 group(분포 이동) OOD를 분리 분석 | 교수님 피드백 3 |
| 2026-11-16 | 데이터는 공식 Waterbirds 사용, 직접 합성 안 함 | 작업량 |
| 2026-11-16 | Train 구성 **B안**: train+val pool에서 ρ 0.95 / 0.5 각 1,678장 | 크기·클래스 수를 맞추면서 데이터를 최대로 확보 |
| 2026-11-16 | 모델: ResNet-18/50 × ρ 0.95/0.5 × seed 0–2 = 12개, seed가 subsample과 초기화를 모두 결정 | K-fold 대신 seed로 데이터 조합 변동까지 반영 |
| 2026-11-16 | 고정 epoch (기본 30, pilot으로 확정), val 없음, test로 결정하지 않음 | 공정 비교, val을 학습에 사용 |
| 2026-11-16 | SRI = 클래스별 \|P(wb\|water) − P(wb\|land)\| 평균 (공식 test 사용) | 합성 없이 측정 가능 (test는 클래스 내 배경 50:50 무작위) |
| 2026-11-16 | 데이터는 WILDS 배포본(CodaLab) 사용 | Stanford 원 배포 서버 접속 불가. `make_splits.py`의 원 논문 그룹별 개수 검사 통과 → 원본과 같은 split |
| 2026-11-16 | conda env는 conda-forge 채널로 생성 | 기본 채널은 Anaconda 이용약관 동의 필요 |
| 2026-11-17 | 결과는 `results/<arch>/`에 작은 파일만 git으로 공유 (checkpoint 제외), 실험 단위로 브랜치를 따서 올림 | 팀 공유·재현용. `scripts/export_results.sh`로 Seraph에서 복사 |
| 2026-11-16 | **epoch 수 확정: 30** | pilot(R18, ρ 0.95/0.5 seed 0, 50 epoch): 두 모델 모두 ~10 epoch에 train acc 1.0, ~20 epoch에 loss 수렴(≈0.002) → 여유 두고 30 |
| 2026-11-19 | [kms78299] 단계 5·6을 ariel 서버에서 팀 checkpoint 12개로 실행 (데이터: WILDS 배포본을 HuggingFace `pangwei/wilds-waterbirds`에서 받음 — CodaLab 주소는 ariel에서 404) | 팀 `splits/*.csv` 8개 파일명·y·place 전부 일치 확인 |
| 2026-11-19 | [kms78299] Attribution 설정: 설명 대상 = 원본 예측 class, Grad-CAM(layer4), IG(픽셀 black, 50 step, 채널 합·부호 유지), RISE(4000 마스크, 7×7, p 0.5, black fill), Random(이미지별 고정 seed) | 팀 계획 "IG/RISE baseline 기본값 고정" |
| 2026-11-19 | [kms78299] Faithfulness 설정: 14×14 grid(16px, 196 patch), 4 patch마다 1점(50점), target softmax 확률, Mean = Waterbirds train+val 평균색, Blur k41 σ8, Inpaint = push-pull 다중해상도 확산, Train-set = 이미지마다 train+val pool의 고정 1장 | 단계 9-3에 적힌 다른 로컬 구현(ImageNet 평균, σ10, normalized conv σ16, train 100장 donor)과 **설정이 다름 → 어느 쪽을 공식으로 할지 결정 필요** |

## 결과

### SRI 검증 (단계 4)
> 해석과 한계, 다음 단계에 주는 의미: [`ANALYSIS_SRI.md`](ANALYSIS_SRI.md)

`python src/aggregate.py` 출력의 표를 붙여넣는다. 상세(run별 지표, 학습 로그, test 예측)는 [`results/resnet18/`](../results/resnet18/README.md).

| arch | rho | n | avg acc | worst-group acc | group gap | SRI |
|---|---|---|---|---|---|---|
| resnet18 | 0.5 | 3 | 0.936 ± 0.001 | 0.787 ± 0.020 | 0.016 ± 0.001 | 0.014 ± 0.001 |
| resnet18 | 0.95 | 3 | 0.811 ± 0.005 | 0.339 ± 0.003 | 0.413 ± 0.009 | 0.410 ± 0.010 |
| resnet50 | 0.5 | 3 | 0.953 ± 0.001 | 0.852 ± 0.006 | 0.005 ± 0.004 | 0.005 ± 0.002 |
| resnet50 | 0.95 | 3 | 0.830 ± 0.006 | 0.541 ± 0.016 | 0.337 ± 0.004 | 0.342 ± 0.007 |

통과 여부: ✅ ResNet-18 통과 (ρ = 0.95가 seed 3개 모두에서 SRI 0.40–0.42 vs clean 0.012–0.015, worst-group 0.34 vs 0.77–0.81)
통과 여부: ✅ ResNet-50 통과 (ρ = 0.95: SRI 0.33–0.35, worst-group 0.52–0.56 vs clean: SRI 0.004–0.007, worst-group 0.85–0.86). 상세: [`results/resnet50/`](../results/resnet50/README.md)

해석 메모:
- shortcut 모델은 같은 클래스에서 배경만 바뀌어도 P(waterbird)가 평균 약 41%p 움직임. clean은 약 1.4%p.
- clean의 worst-group acc(0.79)가 평균(0.94)보다 낮지만 group gap은 0.016으로 작음 → 배경 때문이 아니라 클래스 자체 난이도/불균형(waterbird가 적음) 영향으로 보임.
- **ResNet-50은 같은 ρ에서도 배경 의존이 더 약함** (shortcut SRI 0.342 vs R18 0.410). 12개 모델의 SRI가 4수준(0.005 / 0.014 / 0.342 / 0.410)으로 나뉘어, 단계 7–8에서 SRI–민감도 관계를 경향으로도 분석 가능.

### 단계 5–6 (ariel 실행, kms78299, 11/19–11/21) — 1차, 통계 검정 전
> 상세 설정·파일 설명: [`results/faithfulness/README.md`](../results/faithfulness/README.md) · 그림: `results/faithfulness/figs/` · 코드: `src/attribution.py`, `src/faithfulness.py`, `src/plot_faithfulness.py`, `src/check_ckpts.py`

- **checkpoint 재현 확인:** 팀 `final.pt` 12개를 ariel에서 로드해 test 5,794장 평가 → 팀 `test_preds.csv`와 예측 일치 99.4–99.9%, 지표 차이 ≤ 0.019, SRI 차이 ≤ 0.003 (불일치는 p_wb 0.50–0.64 경계 이미지뿐; Pillow/torchvision 버전 차이로 추정)
- **단계 5:** 12개 모델 × `test_xai800` × {Grad-CAM, IG, RISE, Random} 생성 완료 (map은 용량 때문에 git 제외). IG completeness 상대오차 median 0.06–0.17
- **단계 6:** 12개 모델 × method 4 × baseline 5 × {MIF, LIF} 완료. `results/faithfulness/<run>/auc.csv`(이미지별), `summary.csv`, `summary_all.csv`
- Random 순서 SRG ≈ 0 (−0.002–0.007) → 측정 정상. SRG: RISE·Grad-CAM 0.21–0.42, IG 0.02–0.12
- 순위는 대부분 RISE > Grad-CAM > IG; 변화는 주로 RISE↔Grad-CAM 교체이며 Train-set baseline에서 12개 모두 Grad-CAM 1위
- 기법 3개 Kendall τ 민감도(동점 처리 전 1차 계산)는 shortcut/clean 간 방향이 섞임 (MIF: R18 0.222 vs clean 0.178, R50 0.089 vs 0.178) → 단계 7–8 명세대로 재계산 필요
- 탐색 관찰 (그림 `fig3`, `fig4`): Black → Train-set에서 SRG 하락 폭이 shortcut이 더 큼 (Grad-CAM R18 −0.12 vs −0.05, R50 −0.11 vs −0.03). conflicting 그룹에서 shortcut SRG가 clean보다 높다가 Train-set에서만 비슷해짐

### 단계 7–9 분석 (ariel 실행, kms78299, 11/21) — 1차
> 명세: PROJECT_PLAN "단계 7–10 상세 실행 계획" · 코드: `src/analysis_stage7_9.py` · 결과·정의·파일 설명: [`results/analysis/README.md`](../results/analysis/README.md)

- 7-0 입력 검증 171항목 통과. 순위 민감도 S = 1 − 평균 Kendall τ-b (동점 처리, scipy 대조 검증), bootstrap 4그룹×200장 2,000회
- **8-1 주 결과 (MIF, 전체 800장): H2 미지지.** 평균 D = S_shortcut − S_clean: R18 +0.044 (D>0 seed 1/3, 구간 [−0.04, 0.27]), R50 −0.089 (0/3, [−0.22, 0.00])
- 8-4 SRG 안정화 반대 방향: G = S_MIF − S_SRG 12개 모델 모두 음수 (주 원인: Train-set 에서 Grad-CAM↔RISE 역전)
- 7-2 보조: baseline별 점수 SD 가 Grad-CAM·RISE 에서 shortcut 이 clean 의 1.3–4.2배 (IG 는 0.6–1.0배)
- 9-1 탐색: aligned 그룹 MIF D 가 R18 +0.67, R50 +0.71 (seed 3/3, 구간 0 제외). 기법 간 MIF 차이가 shortcut 에서 0.02 수준으로 작아 근소한 차이의 역전으로 보임
- 9-4 견고성: baseline 하나 제외·공통 정답 subset 에서 D 부호가 바뀜 → 주 결과 약함
- 미완: 9-3 baseline 구현 문서화, 공통 정답 subset bootstrap, 단계 10

### Baseline 별 OOD 정량화 (ariel 실행, kms78299, 11/23) — 탐색
> 코드: `src/baseline_ood.py`, `src/plot_baseline_ood.py` · 방법·결과·한계: [`results/baseline_ood/README.md`](../results/baseline_ood/README.md)

- 같은 무작위 patch 를 25/50/75% 지우고 채우는 값만 바꿔 특징 공간에서 ① OOD 정도(Mahalanobis·KNN, 원본 대비 AUROC) ② 정보 잔존(cosine, 예측 유지) 측정. 모델 12개 + ImageNet ResNet-50 기준
- OOD 순서 (25%, 5종류 모델 모두 동일): **Black > Mean > Train-set > Blur ≈ Inpaint** (AUROC 0.99–1.00 → 0.80–0.91). 정보 잔존은 거의 반대
- H3-1 부분 지지: Black 은 가장 OOD, Blur 는 가장 덜 OOD
- OOD 정도는 순위 역전의 주원인과 불일치: 역전은 Train-set 이 주도, 가장 OOD 인 Black 은 역전 거의 없음
- H3-2: OOD 정도 자체의 shortcut/clean·그룹 간 차이는 작음 (≤0.05). 대신 shortcut 은 conflicting 그룹에서만 Blur·Inpaint 의 예측 유지 이점이 사라짐 (R18 0.60 vs clean 0.84) → 배경 단서 잔존과 분포 이동의 상호작용으로 해석 (피드백 3)

### 마스킹 적응 추가 학습 실험 (ariel 실행, kms78299, 11/23) — 원인 탐색
> 설계·결과·한계: [`results/finetune/README.md`](../results/finetune/README.md) · 코드: `src/finetune_masking.py`, `src/compare_finetune.py`

- R18 checkpoint 6개를 control(원본만) / maskadapt(원본 + 무작위 10–30% 마스킹, baseline 5종)로 5 epoch 추가 학습 (설정은 결과 전에 고정, 두 조건 배치 순서 동일)
- SRI: shortcut 0.409 → control 0.412 / maskadapt 0.377 (clean 과의 차이 유지 → "비슷"). maskadapt shortcut worst-group 0.33 → 0.26
- aligned MIF 순위 불안정성 대부분 유지: shortcut S 0.844 / 0.889 / 0.711, D 는 세 조건 모두 seed 3/3 양수. SRG 는 세 조건 동일
- 고정 순서 진단에서 maskadapt 가 오히려 더 불안정(0.978) → 감소분은 입력 반응보다 설명 변화 쪽
- 판정: **마스킹에 대한 낯섦만으로 기존 순위 역전을 설명하기 어려움.** 전체 800장 MIF 변화는 control 변동(0.22→0.40)과 같은 잡음 수준이라 해석하지 않음

### 진행 중인 실험 (ariel, kms78299, 11/23 제출) — 결과 대기
> 결과가 나오면 이 소절 아래에 결과 소절을 추가한다. 아래는 무엇이 돌아가고 있는지만 기록.

1. **ROAR** (job 499566, 12개 array, 동시 2개) — `src/roar.py`, 분석 `src/roar_analysis.py`
   - 모델 12개(R18·R50 × clean/shortcut × seed 0–2). 원래 모델의 Grad-CAM·IG·RISE·Random 으로 14×14 patch 상위 10/30/50/70% 를
     학습 split(1,678장)과 test_xai800 에서 지우고, **지운 학습 이미지로 처음부터 재학습**(팀 학습과 같은 설정: ImageNet 사전학습, 30 epoch, lr 1e-3)한 뒤 지운 test 정확도 측정
   - 채우는 값: mean (ROAR 원 논문 방식) / trainset. run 당 재학습 33회 (지우기 0% 1회 포함)
   - 볼 것: 기법별 ROAR 점수와 Random 대비, mean ↔ trainset 에서 기법 순위가 바뀌는지와 그 변화의 shortcut/clean 차이, ROAR 순위와 Deletion(MIF) 순위의 일치
   - smoke(job 499555, 학습 16장·1 epoch) 통과
2. **마스킹 적응 추가 학습 — ResNet-50** (job 499570, ROAR 종료 후 자동 시작 `--dependency=afterany:499566`)
   - ResNet-18 실험(`results/finetune/`)과 같은 설계·설정을 R50 6개에 적용 (control / maskadapt, 5 epoch, lr 1e-4, 10–30% 마스킹, baseline 5종)
   - 목적: R18 결론("마스킹 낯섦만으로 순위 역전을 설명하기 어려움")이 아키텍처가 바뀌어도 유지되는지 확인 (원래 결과에서 R18·R50 방향이 달랐던 부분 포함)
   - smoke(job 499549) 통과. `src/compare_finetune.py` 는 R18·R50 을 함께 비교하도록 수정
- 코드 수정: `src/faithfulness.py` — attribution 행을 이미지 번호 대신 파일명으로 찾도록 수정 (attribution 을 일부 이미지만 만든 smoke 에서 생기던 IndexError 해결; 800장 전체 실행 결과는 동일)

### 결과: ROAR · 마스킹 적응 R50 · H3 (ariel, kms78299, 11/24)
> 위 "진행 중인 실험" 소절의 실험 결과. 상세는 각 README.

- **ROAR** ([`results/roar/README.md`](../results/roar/README.md)): 재학습하면 mean ↔ trainset 기법 순위가 12개 중 11개에서 동일 (D_roar = 0) —
  Deletion 의 baseline 의존 순위 변화가 사라짐. ROAR 순위는 대체로 Grad-CAM > IG > RISE 이고 RISE 는 33–83% 만 Random 보다 나음,
  ROAR 와 MIF 순위는 대체로 불일치 (τ −0.33 ~ +0.11). 한계: 지운 모양을 통한 정보 누출 가능성 (F-Fidelity 지적), baseline 2종
- **마스킹 적응 R18+R50** ([`results/finetune_r50/README.md`](../results/finetune_r50/README.md), R18 단독 결과 `results/finetune/` 는 그대로):
  R50 SRI 0.344 → 0.342 / 0.341 (비슷). shortcut aligned MIF 순위 민감도 R50 0.933 → control 0.933 / maskadapt 0.578 (고정 순서 진단도 0.756 으로 감소),
  R18 0.844 → 0.889 / 0.711. 판정: 낯섦이 일부 기여(R50)하나 두 arch 모두 불안정성이 남아 그것만으로는 설명되지 않음
- **H3-1** ([`results/h3/README.md`](../results/h3/README.md), 설계 `docs/H3_DESIGN.md`): clean 모델의 배경을 OOD baseline(Black·Mean·Blur·Inpaint)으로
  가려도 같은 장소 자연 배경 교체보다 더 흔들리지 않음 (R50 Black 만 +0.03), 왜곡은 Train-set (+0.06, 3/3). 틀린 배경 설명은 모든 baseline 에서 Random 보다
  나쁘게 평가됨. 자동 판정: **원인은 OOD 가 아니라 채운 내용** (H3-1 문장 그대로는 미지지)
- **H3-2**: 공통으로 안 보는 칸 진행 기준 통과 (≥2칸 이미지 55–61%). shortcut − clean 반응 차이는 Train-set 에서 두 arch 모두 +0.07 (3/3),
  R50 만 Mean·Inpaint·Black 에서 +0.02–0.03. 자동 판정: **부분적**. 한계: 공통 칸 전체를 자연 배경으로 바꿔도 Δp 0.08–0.13 (칸 단위 판정은 근사)
- 종합: Deletion 평가를 흔드는 것은 baseline 의 OOD 정도보다 **채운 자리에 들어가는 내용**(특히 다른 이미지의 배경·새를 넣는 Train-set)으로 보이며,
  shortcut 모델의 추가 민감도도 주로 그 경우에 나타남

### 기법 확장 견고성 분석: SmoothGrad · LIME 추가 (ariel, kms78299, 11/25)
> 설계(실행 전 고정) `docs/METHODS_EXT_DESIGN.md` · 결과 [`results/methods_ext/README.md`](../results/methods_ext/README.md). **주 결과는 기법 3개 분석 그대로.**

- 목적: 기법 3개 순위 지표가 거칠어 seed 마다 D 부호가 바뀌던 문제 → 기법 5개(순위 쌍 10개)로 해상도를 높여 확인
- 확인용: 같은 평가 결과에서 기법 3개로 다시 계산한 값이 기존 주 결과와 정확히 일치
- 기법 성능(SRG): LIME 0.28–0.44 (RISE 와 최상위), SmoothGrad 0.21–0.25 (baseline 에 거의 둔감). LIME 도 Train-set 에서 하락
- 전체 800장 MIF D: R18 0.00 [−0.07, 0.08], R50 −0.08 [−0.15, 0.05], seed 별 방향 섞임 → **판정: 주 분석(H2) 미지지를 뒷받침**
- aligned MIF D: R18 +0.39, R50 +0.41, 두 arch 모두 seed 3/3·구간 0 제외 → 기법 3개 때보다 일관되게 재현. SRG 가 더 불안정(G < 0)한 것도 동일

### LIF 지표 확인 (ariel, kms78299, 11/25)
> 결과 [`results/lif_check/README.md`](../results/lif_check/README.md). 저장된 곡선으로 다시 계산만 함. **주 결과는 그대로.**

- `src/analysis_stage7_9.py` 에 `--metrics` 추가 (기본 `mif srg` 출력은 기존과 동일)
- LIF 순위 민감도 D: 기법 5개에서 두 arch 모두 전체·aligned·conflicting 음수 (shortcut 순위가 더 안정). 기법 3개는 R18 음수, R50 섞임
- 점수 변동은 Grad-CAM·RISE 에서 shortcut 이 더 큼 → H2 방향은 지표 선택(MIF/LIF/SRG)에 의존

### ρ 수준 추가 실험: ρ 0.75 · 0.99 (ariel, kms78299, 11/25)
> 설계(실행 전 고정) `docs/RHO_LEVELS_DESIGN.md` · 결과 [`results/rho_levels/README.md`](../results/rho_levels/README.md). **추가 실험, 기존 주 분석은 그대로.**

- 새 모델 12개 (R18·R50 × ρ 0.75·0.99 × seed 0–2), split 은 팀 `make_pair` 로 생성 (기존 split 완전 재현 확인). 기존 12개와 합쳐 24개로 경향 분석
- SRI: R18 0.014 / 0.121 / 0.409 / 0.607, R50 0.005 / 0.074 / 0.344 / 0.573 (ρ 0.5 / 0.75 / 0.95 / 0.99)
- 주 분석(기법 3개, MIF, 전체): R18 증가 (3/3, 기울기 +0.52 [0.28, 0.82]), R50 아님 (1/3, [−0.51, 0.25]) → **순위 단조 경향 확인 안 됨 (H2 미지지와 일관)**
- 보조: 점수 변동은 Grad-CAM·RISE·LIME·SmoothGrad × 두 arch × 세 지표 모두 의존도와 함께 증가 (3/3). aligned MIF 순위 민감도도 두 arch 모두 증가 (3/3, 구간 0 제외).
  LIF 는 R18 감소 / R50 섞임 → 판정: **점수 수준·특정 조건(aligned, MIF)에서만 나타나는 효과**

## 실행 기록 (Slurm job)

| 날짜 | job ID | 내용 | 결과 / 메모 |
|---|---|---|---|
| 11/16 | 220250 | pilot: R18 ρ0.95/0.5 seed0, 50 epoch, 평가 생략 | epoch당 2–3초, ~20 epoch 수렴 → 30 확정 |
| 11/16 | 220260 | R18 본 학습 6개 (30 epoch) + SRI | 전부 정상 종료, SRI 검증 통과 |
| 11/17 | 220467 | R50 본 학습 6개 (30 epoch) + SRI | 전부 정상 종료, SRI 검증 통과 |
| 11/19 | 495072 (ariel) | 팀 checkpoint 12개 로드 + test 평가·팀 결과 대조 | 예측 일치 99.4–99.9% |
| 11/19 | 495107, 495108 (ariel) | 단계 5 attribution smoke(16장) / 본 실행 12개 | 정상 종료, RISE R18 ~7분·R50 ~20분/모델 |
| 11/19 | 495185 (ariel) | 단계 5 vis.png 다시 그리기 (map 재사용) | 정상 종료 |
| 11/19 | 495236, 495252 (ariel) | 단계 6 faithfulness smoke(16장) / 본 실행 12개 | 정상 종료 |
| 11/23 | 498742, 498743 (ariel) | baseline OOD 정량화 smoke(16장) / 본 실행 13개 (모델 12 + ImageNet R50) | 정상 종료 |
| 11/23 | 498763 (ariel) | 마스킹 적응 추가 학습 12개 (R18 6개 × control/maskadapt) + SRI 재평가 + attribution + faithfulness(새·고정 순서) | 정상 종료 |
| 11/23 | 499555 (ariel) | ROAR smoke (R18 shortcut seed0, 학습 16장·1 epoch) | 정상 종료 |
| 11/23 | 499566 (ariel) | ROAR 본 실행 12개 (R18·R50) | **진행 중** |
| 11/23 | 499530, 499549 (ariel) | R50 마스킹 적응 smoke (499530 은 faithfulness 단계 IndexError → 코드 수정 후 499549 통과) | 정상 종료 |
| 11/23 | 499570 (ariel) | R50 마스킹 적응 본 실행 12개 (ROAR 종료 후 시작) | **대기 중** |
| 11/24 | 499566 (ariel) | ROAR 본 실행 12개 | **완료** (위 행 "진행 중" 의 결과) |
| 11/24 | 499570 (ariel) | R50 마스킹 적응 본 실행 12개 | **완료** (위 행 "대기 중" 의 결과) |
| 11/23 | 499615, 499624 (ariel) | H3 smoke (499615 은 npz 읽기 오류 → 수정 후 499624 통과) | 정상 종료 |
| 11/24 | 499638, 499649, 499669 (ariel, 분할 제출) | H3 본 실행 6쌍 (실험 A·B, H3-2) | **완료** |
| 11/25 | 501640 (smoke), 501641 (ariel) | 기법 확장: SmoothGrad·LIME attribution + 6개 기법 faithfulness (모델 12개) | 정상 종료 |
| 11/25 | 501842 (smoke), 501850 (ariel) | ρ 0.75·0.99 새 모델 12개: 학습 + SRI + attribution 6종 + faithfulness 6종 | 정상 종료 |

## 이슈 / 메모
- Seraph GPU 노드: pypi/pytorch는 접속되지만 nlp.stanford.edu는 DNS 실패. 데이터는 Mac에서 받아 `scp ... moana:`로 업로드함. → 절차를 [`DATASET.md`](DATASET.md)와 `scripts/prepare_data_local.sh`로 정리 (11/19)
- 처음 가는 노드에서 `conda: command not found` → `source /data/$USER/anaconda3/etc/profile.d/conda.sh` 후 activate (sbatch 스크립트는 이미 이렇게 처리).
- Seraph 저장소는 repo-local git user 설정(`git config user.name/email`)이 필요했음.
- 11/16 19:45경부터 moana 외부 접속(30080) 불가: TCP 연결은 되지만 SSH banner가 오지 않음 (서버 쪽 SSH 응답 없음). 11/17에 복구됨. 장애 전 R50은 제출되지 않았음.
- Seraph에서 `git fetch` 인증 실패 (저장된 토큰 만료). VSCode 터미널 인증 또는 SSH 키로 해결 필요.
- [kms78299] 단계 5·6 구현이 두 벌 존재: `exp/faithfulness` 원격 브랜치(16e4788 이후, ariel 실행 결과)와 PROJECT_PLAN 단계 9-3이 검토한 로컬 구현(ecd5158/0e37386). baseline 설정이 달라 공식 구현·결과를 하나로 정해야 함. 같은 브랜치 이름으로 push 시 충돌 주의.
- [kms78299] ariel 결과는 팀 Seraph 예측과 이미지 단위로 완전히 같지 않음(위 재현 확인) → ariel 분석은 ariel 예측 기준으로 일관되게 사용.
