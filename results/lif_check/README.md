# LIF 지표로 다시 본 단계 7–9 (확인용)

- 날짜: 2026-11-25 · 실행: ariel (CPU, 새 학습·평가 없음) — 이미 저장된 faithfulness 곡선에서 **LIF(= Insertion, 높을수록 좋음)** 기준으로 단계 7–9 분석을 다시 계산
- 목적: 주 분석은 MIF·SRG 였다. LIF 도 저장돼 있었으므로 같은 결론이 나오는지 확인. **주 결과는 바뀌지 않는다.**
- 코드: `src/analysis_stage7_9.py --metrics lif ...` (`--metrics` 옵션 추가. 기본값 `mif srg` 출력은 기존과 바이트 단위로 동일함을 확인)

## 폴더
모든 폴더는 `--metrics mif lif srg` 로 계산 (MIF·SRG 값은 기존 결과와 같고, LIF 가 추가된 것)

| 폴더 | 입력 |
|---|---|
| `main/` | 단계 6 (기법 3개, 모델 12개) |
| `methods5/` | 기법 확장 (기법 5개, `--xai_methods gradcam ig rise smoothgrad lime`) |
| `ft_control/`, `ft_maskadapt/` (+ `_fixed`) | 마스킹 적응 추가 학습 (R18·R50, 새 순서 / 기존 순서 고정) |

각 폴더: `bootstrap_summary.csv`, `sensitivity.csv`, `paired_comparison.csv`, `score_variation.csv`, `validation_report.md`(모두 통과), `figs/`

## 결과: LIF 순위 민감도 D = S_shortcut − S_clean (arch 평균 [bootstrap 95%], seed 중 D>0 개수)
| | R18 | R50 |
|---|---|---|
| 기법 3개, 전체 | −0.13 [−0.31, 0.04], 0/3 | +0.09 [0.04, 0.27], 2/3 |
| 기법 3개, aligned | −0.13, 0/3 | −0.13, 0/3 |
| **기법 5개, 전체** | **−0.25 [−0.32, −0.15], 0/3** | **−0.20 [−0.24, −0.04], 0/3** |
| 기법 5개, aligned / conflicting | −0.19 / −0.27, 0/3 | −0.13 / −0.20, 0/3 |

- LIF 에서는 shortcut 모델의 순위가 오히려 **더 안정적** (기법 5개에서 모든 범위 음수). MIF aligned 에서 보이던 shortcut 의 불안정성은 LIF 에서 나타나지 않음
- 반면 점수 변동(SD)은 Grad-CAM·RISE 에서 shortcut 이 더 큼 → 점수는 더 흔들리지만 순위는 덜 흔들림
- 마스킹 적응 후(`ft_maskadapt_fixed`)에는 D 가 0 근처로 섞임

## 해석
지표에 따라 shortcut 의 순위 민감도 방향이 달라진다 (MIF aligned: 증가, LIF: 감소, SRG: 섞임). → **H2 의 "shortcut 모델 평가가 더 불안정하다"는 지표 선택에 의존**하며, 일반적 결론으로 볼 수 없다. ρ 4수준 실험(`results/rho_levels/`)에서도 같은 방향이 반복됨.
