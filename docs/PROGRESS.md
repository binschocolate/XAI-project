# 진행 현황 (항상 최신으로 유지)

> 작업할 때마다 이 파일을 업데이트한다. 새로 합류하거나 오랜만에 볼 때 **"지금 어디까지 했고 다음에 뭘 하나"**를 여기서 바로 알 수 있어야 한다.
> 큰 방향은 [`PROJECT_PLAN.md`](PROJECT_PLAN.md), Seraph 명령은 `SERAPH_GUIDE.md`(로컬 전용).

**마지막 업데이트:** 2026-11-17

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
| 3c/4 | ResNet-50 학습 + SRI | ⏳ **다음** (제출 여부 확인 필요, 아래 이슈 참고) |
| 5 | Attribution 생성 | ⬜ |
| 6 | Faithfulness 평가 (baseline 5종) | ⬜ |
| 7–9 | 민감도, 주 분석, 원인 분석 | ⬜ |
| 10 | 발표 / 보고서 | ⬜ |

## 다음에 할 일 (위에서부터)
1. Seraph 접속 복구 후 ResNet-50 상태 확인 (`squeue -u $USER`, `ls /data/$USER/results/XAI-project/models/`). 없으면 `EPOCHS=30 ARCH=resnet50 sbatch scripts/train.sbatch` → `src/aggregate.py` → `bash scripts/export_results.sh resnet50` → 아래 결과 표와 `results/resnet50/README.md` 작성
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

## 결과

### SRI 검증 (단계 4)
`python src/aggregate.py` 출력의 표를 붙여넣는다. 상세(run별 지표, 학습 로그, test 예측)는 [`results/resnet18/`](../results/resnet18/README.md).

| arch | rho | n | avg acc | worst-group acc | group gap | SRI |
|---|---|---|---|---|---|---|
| resnet18 | 0.5 | 3 | 0.936 ± 0.001 | 0.787 ± 0.020 | 0.016 ± 0.001 | 0.014 ± 0.001 |
| resnet18 | 0.95 | 3 | 0.811 ± 0.005 | 0.339 ± 0.003 | 0.413 ± 0.009 | 0.410 ± 0.010 |
| resnet50 | 0.5 | | | | | |
| resnet50 | 0.95 | | | | | |

통과 여부: ✅ ResNet-18 통과 (ρ = 0.95가 seed 3개 모두에서 SRI 0.40–0.42 vs clean 0.012–0.015, worst-group 0.34 vs 0.77–0.81)

해석 메모:
- shortcut 모델은 같은 클래스에서 배경만 바뀌어도 P(waterbird)가 평균 약 41%p 움직임. clean은 약 1.4%p.
- clean의 worst-group acc(0.79)가 평균(0.94)보다 낮지만 group gap은 0.016으로 작음 → 배경 때문이 아니라 클래스 자체 난이도/불균형(waterbird가 적음) 영향으로 보임.

## 실행 기록 (Slurm job)

| 날짜 | job ID | 내용 | 결과 / 메모 |
|---|---|---|---|
| 11/16 | 220250 | pilot: R18 ρ0.95/0.5 seed0, 50 epoch, 평가 생략 | epoch당 2–3초, ~20 epoch 수렴 → 30 확정 |
| 11/16 | 220260 | R18 본 학습 6개 (30 epoch) + SRI | 전부 정상 종료, SRI 검증 통과 |

## 이슈 / 메모
- Seraph GPU 노드: pypi/pytorch는 접속되지만 nlp.stanford.edu는 DNS 실패. 데이터는 Mac에서 받아 `scp ... moana:`로 업로드함.
- 처음 가는 노드에서 `conda: command not found` → `source /data/$USER/anaconda3/etc/profile.d/conda.sh` 후 activate (sbatch 스크립트는 이미 이렇게 처리).
- Seraph 저장소는 repo-local git user 설정(`git config user.name/email`)이 필요했음.
- 11/16 19:45경부터 moana 외부 접속(30080) 불가: TCP 연결은 되지만 SSH banner가 오지 않음 (서버 쪽 SSH 응답 없음). ResNet-50 제출 여부와 결과는 복구 후 확인 필요.
