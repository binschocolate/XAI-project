# H3-1 · H3-2 검증 결과

- 날짜: 2026-11-24 · 실행: ariel · 설계(결과 전 고정, 커밋 f7e5385): [`docs/H3_DESIGN.md`](../../docs/H3_DESIGN.md)
- 코드: `src/h3_prepare.py`, `src/h3_experiment.py`, `src/faithfulness.py`(oracle map), `src/h3_analysis.py`
- "안 쓰는 부분"을 XAI·OOD 마스킹 없이 정의: H3-1 = clean 모델의 배경 (CUB segmentation), H3-2 = clean·shortcut 이 **모두** 안 보는 4×4 칸
  (물·땅 자연 배경 조각 8장으로 바꿔도 |Δp| < 0.02). OOD 정도 = ImageNet ResNet-50 (Waterbirds 미학습) Mahalanobis AUROC
- 준비 확인: segmentation 과 Waterbirds 이미지 크기 전부 일치 (`prep_report.json`, seg_resized = 0)

## 파일
- `summary.md` (자동 판정 포함), `expA.csv`, `expB.csv`, `h32_selection.csv`, `h32.csv`, `figs/fig_h3_overview.png`
- `pairs/<arch>_seed<k>/{raw.npz, config.json}` 이미지별 원자료, `oracle_faith/<clean run>/` 실험 B faithfulness 요약

## H3-1 (clean 6개)
**전제:** 반대 장소 배경 교체 Δp 0.036–0.041 (기준 0.05 이하) → clean 은 배경을 거의 안 씀. 단 같은 장소 교체도 Δp 0.035–0.045

**실험 A — 인공물 = Δp(baseline 으로 채움) − Δp(같은 장소 자연 배경 교체)**

| | Black | Mean | Blur | Inpaint | Train-set |
|---|---|---|---|---|---|
| R18 | −0.006 | −0.013 | −0.012 | −0.007 | **+0.062** (3/3) |
| R50 | **+0.028** (3/3) | −0.003 | 0.000 | +0.001 | **+0.065** (3/3) |

ρ(OOD, 인공물) −0.4 ~ 0 → OOD baseline 으로 안 쓰는 배경을 가려도 자연 교체보다 더 흔들리지 않음 (R50 Black 만 소폭). 왜곡은 Train-set

**실험 B — 틀린 설명(background map)이 Random 보다 좋게 평가되는가:** 모든 baseline·모델에서 아니오
(부당한 점수 = MIF(random) − MIF(background) −0.18 ~ −0.25, 구간이 0 보다 큰 seed 0개)

**판정 (자동, 사전 기준):** 인공물은 있으나 OOD 순서와 일치하지 않음 → **원인은 OOD 가 아니라 채운 내용.** H3-1 은 문장 그대로는 지지되지 않음

## H3-2 (clean–shortcut 쌍 6개)
**단계 1:** 공통 칸 ≥ 2 인 이미지 55–61% (진행 기준 50% 통과), 평균 6.6–7.8 칸. 안 보는 칸 비율 clean 66–76%, shortcut 53–57%.
위치는 가장자리·구석(≈0.5) > 가운데(0.27–0.34)

**단계 2 — diff = Δp_shortcut − Δp_clean (공통 칸, τ = 0.02)**

| | Black | Mean | Blur | Inpaint | Train-set |
|---|---|---|---|---|---|
| R18 | −0.007 | +0.001 | −0.024 | 0.000 | **+0.068** (3/3) |
| R50 | +0.024 (2/3) | +0.031 (3/3) | −0.001 | +0.028 (3/3) | **+0.073** (3/3) |

- Train-set 에서 두 arch 모두 shortcut 이 더 크게 반응 (다른 이미지의 배경 정보가 들어오는 것과 관련된 것으로 보임)
- R50 만 Mean·Inpaint·Black 에서 소폭 더 민감, R18 은 아님. OOD 순서와 무관 (Blur ≈ 0, 비슷한 OOD 의 Inpaint +0.03)

**판정 (자동, 사전 기준):** 부분적

## 한계
- **"안 보는 칸"은 근사:** 공통 칸 전체를 한꺼번에 자연 배경으로 바꾸면 Δp 0.08–0.13 → 칸 단위로는 안 봐도 함께 바꾸면 영향이 있음
- clean 도 같은 장소 배경 교체에 Δp ≈ 0.04 반응 (배경을 완전히 무시하지는 않음)
- Train-set donor 에 다른 이미지의 새가 들어갈 수 있음; OOD AUROC 는 원본 test 대비 상대값 (배경 전체를 바꾼 Train-set 은 0.37 로 원본보다 "덜 OOD")
