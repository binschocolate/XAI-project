# 진행 현황 (항상 최신으로 유지)

> 작업할 때마다 이 파일을 업데이트한다. 새로 합류하거나 오랜만에 볼 때 **"지금 어디까지 했고 다음에 뭘 하나"**를 여기서 바로 알 수 있어야 한다.
> 큰 방향은 [`PROJECT_PLAN.md`](PROJECT_PLAN.md), Seraph 명령은 `SERAPH_GUIDE.md`(로컬 전용).

**마지막 업데이트:** 2026-11-16

---

## 지금 상태 한눈에

| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | 실험 설계 확정 (B안, 모델 12개) | ✅ 완료 |
| 1 | 저장소 / 코드 / Seraph 스크립트 준비 | ✅ 완료 |
| 1 | Seraph 환경 세팅 + 데이터 다운로드 | ⏳ 다음 |
| 2 | train split 생성 + git 커밋 | ⬜ |
| 3a | pilot 학습으로 epoch 수 확정 | ⬜ |
| 3b | ResNet-18 학습 (6개) | ⬜ |
| 4 | SRI 검증 (ResNet-18) | ⬜ ← **현재 목표** |
| 3c/4 | ResNet-50 학습 + SRI | ⬜ |
| 5 | Attribution 생성 | ⬜ |
| 6 | Faithfulness 평가 (baseline 5종) | ⬜ |
| 7–9 | 민감도, 주 분석, 원인 분석 | ⬜ |
| 10 | 발표 / 보고서 | ⬜ |

## 다음에 할 일 (위에서부터)
1. Seraph에 저장소 clone → `scripts/setup_env.sh`, `scripts/download_data.sh` 실행 (`SERAPH_GUIDE.md` 1–3단계)
2. `src/make_splits.py` 실행 → 출력된 그룹별 개수 확인 → `splits/` 커밋
3. pilot 제출 → `train_log.csv`에서 train loss/acc가 평평해지는 epoch 확인 → 아래 "결정 기록"에 epoch 수 기록
4. ResNet-18 본 학습 제출 → `src/aggregate.py`로 SRI 표 생성 → 아래 "결과"에 붙여넣기

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
| | epoch 수 확정: __ | pilot 결과 |

## 결과

### SRI 검증 (단계 4)
`python src/aggregate.py` 출력의 표를 붙여넣는다.

| arch | rho | n | avg acc | worst-group acc | group gap | SRI |
|---|---|---|---|---|---|---|
| | | | | | | |

통과 여부: ⬜ (ρ = 0.95가 seed 3개 모두에서 SRI·group gap 높고 worst-group acc 낮음)

## 실행 기록 (Slurm job)

| 날짜 | job ID | 내용 | 결과 / 메모 |
|---|---|---|---|
| | | | |

## 이슈 / 메모
- (막힌 점, 이상한 결과, 나중에 확인할 것)
