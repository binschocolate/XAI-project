# ROAR (RemOve And Retrain) 결과

- 날짜: 2026-11-24 · 실행: ariel (job 499566) · 코드: `src/roar.py`, 분석 `src/roar_analysis.py`
- 설정: 모델 12개(R18·R50 × clean/shortcut × seed 0–2). 원래 모델의 Grad-CAM·IG·RISE·Random 으로 14×14 patch 상위 10/30/50/70% 를
  학습 split(1,678장)과 test_xai800 에서 지우고, **지운 학습 이미지로 ImageNet 사전학습부터 재학습**(30 epoch, lr 1e-3, 팀 학습과 동일)한 뒤
  지운 test 정확도 측정. 채우는 값 mean / trainset. run 당 재학습 33회 (지우기 0% 포함)
- ROAR 점수 = 10–70% 에서의 test 정확도 평균 (낮을수록 중요한 곳을 잘 찾은 설명)

## 파일
- `summary.md` 요약표, `roar_all.csv` 모든 조건, `roar_scores.csv` run×method×baseline 점수, `ranks.csv` 순위·τ
- `runs/<run>/{roar.csv, config.json}`, `figs/fig_roar_curves_<arch>.png`

## 결과 (seed 평균)
1. **재학습하면 baseline 에 따른 순위 변화가 거의 사라짐:** mean ↔ trainset 의 기법 순위가 12개 중 11개에서 동일 (τ = 1; 예외 R18 clean seed2).
   D_roar(shortcut − clean) = 0. Deletion 에서 보인 baseline 의존 순위 변화와 대조적
2. **ROAR 순위는 대체로 Grad-CAM > IG > RISE.** Grad-CAM·IG 는 모든 run 에서 Random 보다 낫지만 **RISE 는 33–83% 만 Random 보다 나음**
3. **ROAR 와 Deletion(MIF) 순위는 대체로 불일치:** τ(ROAR, MIF) −0.33 ~ +0.11 (예외 R50 clean·trainset 0.56). Deletion 에서는 RISE 가 1위였음
4. trainset 채움이 mean 보다 모든 방법(Random 포함)에서 정확도를 더 낮춤

| ROAR 점수 (mean / trainset) | Grad-CAM | IG | RISE | Random |
|---|---|---|---|---|
| R18 clean | 0.608 / 0.551 | 0.703 / 0.601 | 0.731 / 0.612 | 0.743 / 0.670 |
| R18 shortcut | 0.557 / 0.528 | 0.586 / 0.551 | 0.663 / 0.585 | 0.613 / 0.592 |
| R50 clean | 0.629 / 0.564 | 0.744 / 0.627 | 0.727 / 0.630 | 0.767 / 0.692 |
| R50 shortcut | 0.582 / 0.531 | 0.601 / 0.557 | 0.661 / 0.580 | 0.641 / 0.607 |

지우기 0% 재학습 정확도: R18 clean 0.892 / shortcut 0.752, R50 clean 0.920 / shortcut 0.796

## 한계
- baseline 2종, 기법 3개라 순위 지표가 거침
- ROAR 는 **지운 모양 자체가 정보를 흘릴 수 있음** (F-Fidelity, ICLR 2025 지적) — 예: Grad-CAM 처럼 새 모양으로 지우면 모양만으로 클래스 추정 가능
- 조건당 재학습 1회 (seed 3개 모델이 반복 역할)
