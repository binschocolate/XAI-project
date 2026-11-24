# 기법 확장 견고성 분석 — SmoothGrad · LIME 추가 (기법 5개)

- 날짜: 2026-11-25 · 실행: ariel (job 501641, 모델 12개) · 설계(실행 전 고정): [`docs/METHODS_EXT_DESIGN.md`](../../docs/METHODS_EXT_DESIGN.md)
- **주 결과는 기법 3개 분석(`results/analysis/`) 그대로다.** 이 폴더는 순위 지표의 해상도(기법 3개 → 순위 쌍 3개)를 높여 본 사후 견고성 분석.
- 코드: `src/attribution.py`(SmoothGrad·LIME 추가, 기존 기법 그대로), `src/faithfulness.py`(6개 기법), `src/analysis_stage7_9.py --xai_methods ...`
- 설정: SmoothGrad 노이즈 50개·σ=0.15×입력 범위, |평균 gradient| 채널 합 / LIME SLIC(n_segments=50 요청 → **실제 이미지당 중간값 32개**), 샘플 1,000개,
  꺼진 superpixel 은 평균색, cosine kernel 0.25 가중 Ridge. 나머지(이미지 800장, baseline 5종, 14×14 patch, target)는 단계 5·6 과 동일

## 파일
- `analysis/` — 기법 5개 단계 7–9 분석 (`validation_report.md` 171항목 통과, sensitivity, paired_comparison, bootstrap_summary …, 그림)
- `analysis_3methods_check/` — 같은 평가 결과에서 기법 3개만으로 다시 계산한 확인용 → **기존 주 결과와 정확히 일치**
- `faithfulness/<run>/` — run 별 6개 기법 faithfulness 요약, `attribution_config/<run>.json` — attribution 설정·소요 시간
- `figs/vis_smoothgrad_lime_resnet18_{shortcut,clean}_seed0.png` — 예시 map
- 이미지별 auc.csv·곡선·map 원본은 용량 때문에 ariel `outputs/faithfulness_m6/`, `outputs/attributions/` 에만 있음

## 결과 (seed 평균)
### 기법 성능 (SRG)
| | Black | Mean | Blur | Inpaint | Train-set |
|---|---|---|---|---|---|
| LIME (R18 / R50) | 0.38 / 0.35 | 0.43 / 0.37 | 0.40 / 0.38 | 0.44 / 0.39 | 0.28 / 0.28 |
| SmoothGrad | 0.21 / 0.25 | 0.21 / 0.25 | 0.21 / 0.25 | 0.22 / 0.25 | 0.21 / 0.24 |
| RISE (참고) | 0.40 / 0.36 | 0.39 / 0.35 | 0.38 / 0.38 | 0.42 / 0.39 | 0.24 / 0.25 |

- MIF 순위는 대부분 LIME > RISE > Grad-CAM > IG·SmoothGrad
- SmoothGrad 는 baseline 에 거의 둔감 (0.21–0.25). LIME 도 Train-set 에서 크게 하락 (기존 기법과 같은 패턴)

### 순위 민감도 D = S_shortcut − S_clean (기법 5개, 순위 쌍 10개)
| | seed 0 / 1 / 2 | 평균 [이미지 bootstrap 95%] |
|---|---|---|
| MIF, 전체 800장 — R18 | 0.00 / −0.12 / +0.12 | 0.00 [−0.07, 0.08] |
| MIF, 전체 800장 — R50 | −0.16 / −0.08 / 0.00 | −0.08 [−0.15, 0.05] |
| **MIF, aligned — R18** | **+0.36 / +0.44 / +0.36** | **+0.39 [0.25, 0.44]** |
| **MIF, aligned — R50** | **+0.36 / +0.40 / +0.48** | **+0.41 [0.28, 0.52]** |
| SRG, conflicting | R18 0 ~ −0.28, R50 −0.20 ~ −0.32 | 음수 (shortcut 이 더 안정) |

- G = S_MIF − S_SRG 는 모든 모델에서 음수 (SRG 가 순위를 더 흔듦, 기법 3개와 같은 방향)
- 점수 변동 배율(shortcut/clean, SRG): LIME 1.36–2.25 (6/6 > 1), SmoothGrad 0.32–3.39 (섞임)

## 판정 (사전 기준)
**기법 5개에서도 전체 800장의 D 방향이 섞이고 0 근처 → 지표를 촘촘히 해도 shortcut 의 순위 불안정성이 일관되게 크지 않음 → 주 분석(H2) 미지지를 뒷받침.**
반면 aligned 그룹에서 shortcut 이 더 불안정하다는 결과는 기법 5개에서 두 arch 모두 seed 3/3, 구간이 0 제외로 더 일관되게 재현됨.

## 한계
- 사후 추가 분석. LIME superpixel 수가 요청(50)보다 적음(중간값 32). SmoothGrad 는 절댓값 map 이라 부호 정보 없음
