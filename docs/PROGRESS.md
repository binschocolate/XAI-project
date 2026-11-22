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

## 이슈 / 메모
- Seraph GPU 노드: pypi/pytorch는 접속되지만 nlp.stanford.edu는 DNS 실패. 데이터는 Mac에서 받아 `scp ... moana:`로 업로드함. → 절차를 [`DATASET.md`](DATASET.md)와 `scripts/prepare_data_local.sh`로 정리 (11/19)
- 처음 가는 노드에서 `conda: command not found` → `source /data/$USER/anaconda3/etc/profile.d/conda.sh` 후 activate (sbatch 스크립트는 이미 이렇게 처리).
- Seraph 저장소는 repo-local git user 설정(`git config user.name/email`)이 필요했음.
- 11/16 19:45경부터 moana 외부 접속(30080) 불가: TCP 연결은 되지만 SSH banner가 오지 않음 (서버 쪽 SSH 응답 없음). 11/17에 복구됨. 장애 전 R50은 제출되지 않았음.
- Seraph에서 `git fetch` 인증 실패 (저장된 토큰 만료). VSCode 터미널 인증 또는 SSH 키로 해결 필요.
- [kms78299] 단계 5·6 구현이 두 벌 존재: `exp/faithfulness` 원격 브랜치(16e4788 이후, ariel 실행 결과)와 PROJECT_PLAN 단계 9-3이 검토한 로컬 구현(ecd5158/0e37386). baseline 설정이 달라 공식 구현·결과를 하나로 정해야 함. 같은 브랜치 이름으로 push 시 충돌 주의.
- [kms78299] ariel 결과는 팀 Seraph 예측과 이미지 단위로 완전히 같지 않음(위 재현 확인) → ariel 분석은 ariel 예측 기준으로 일관되게 사용.
