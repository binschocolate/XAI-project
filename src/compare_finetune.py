"""마스킹 적응 추가 학습 실험 비교 (CPU). original / control / maskadapt 를 한 표로.

입력
  original : models/<run>/metrics_ariel.json, outputs/analysis/sensitivity.csv
  control, maskadapt : models_ft/<cond>/<run>/metrics.json (eval_sri),
                       outputs/ft/<cond>/analysis/sensitivity.csv        (새 attribution 으로 평가 = 최종 결과)
                       outputs/ft/<cond>/analysis_fixed/sensitivity.csv  (원래 모델 attribution·target 고정 = 보조 진단)
출력 (results/finetune/)
  shortcut_check.csv      조건별 avg acc, worst-group acc, group gap, SRI   (단계 2: shortcut 의존성 유지 확인)
  sensitivity_compare.csv 조건 × 평가방식 × 지표 × scope 의 순위 민감도 S, FlipRate
  summary.md              판정용 요약표
  figs/fig_finetune_sensitivity.png
해석 기준 (사전 고정, 실험 설계 문서):
  SRI 비슷 + maskadapt 에서 역전 감소 → 마스킹 입력에 대한 미숙한 반응이 불안정성에 기여했을 가능성
  SRI 비슷 + 역전 유지             → 마스킹에 대한 낯섦만으로 기존 역전을 설명하기 어려움
  SRI 크게 변함                    → shortcut 의존성 변화가 섞여 원인 분리 어려움
사용: python src/compare_finetune.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RUNS = [f"resnet18_{r}_seed{s}" for r in ["rho050", "rho095"] for s in range(3)]
CONDS = ["original", "control", "maskadapt"]
OUT = REPO / "results" / "finetune"
KEYS = ["avg_acc", "worst_group_acc", "group_gap", "sri"]


def metrics_path(cond, run):
    return REPO / "models" / run / "metrics_ariel.json" if cond == "original" else \
        REPO / "models_ft" / cond / run / "metrics.json"


def sens_path(cond, ev):
    if cond == "original":
        return REPO / "outputs" / "analysis" / "sensitivity.csv"
    return REPO / "outputs" / "ft" / cond / ("analysis" if ev == "new_attr" else "analysis_fixed") / "sensitivity.csv"


def main():
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    # 단계 2: shortcut 의존성
    rows = []
    for cond in CONDS:
        for run in RUNS:
            p = metrics_path(cond, run)
            if p.exists():
                m = json.loads(p.read_text())
                rows.append(dict(cond=cond, run=run, model="shortcut" if "rho095" in run else "clean",
                                 seed=int(run[-1]), **{k: m[k] for k in KEYS}))
    sc = pd.DataFrame(rows)
    sc.to_csv(OUT / "shortcut_check.csv", index=False)

    # 단계 3: 순위 민감도
    parts = []
    for cond in CONDS:
        for ev in (["new_attr"] if cond == "original" else ["new_attr", "fixed_order"]):
            p = sens_path(cond, ev)
            if not p.exists():
                print(f"[없음] {p}")
                continue
            s = pd.read_csv(p)
            s = s[s.run.isin(RUNS)].assign(cond=cond, eval=ev)
            parts.append(s[["cond", "eval", "run", "model", "seed", "metric", "scope", "S", "flip_rate", "n_valid_pairs"]])
    sv = pd.concat(parts)
    sv.to_csv(OUT / "sensitivity_compare.csv", index=False)

    # 요약
    lines = ["# 마스킹 적응 추가 학습 실험 요약 (ResNet-18, seed 0–2 평균)\n",
             "## 1. Shortcut 의존성 유지 확인 (test 5,794장)\n",
             "```\n" + sc.groupby(["model", "cond"])[KEYS].agg(["mean", "std"]).round(3).to_string() + "\n```\n"]
    for ev in ["new_attr", "fixed_order"]:
        d = sv[(sv["eval"] == ev) | (sv.cond == "original")]
        t = d[d.scope.isin(["all", "aligned", "conflicting"])].pivot_table(
            index=["metric", "scope", "model"], columns="cond", values="S", aggfunc="mean")
        t = t[[c for c in CONDS if c in t.columns]]
        name = "새 attribution 으로 평가 (최종 결과)" if ev == "new_attr" else "원래 attribution·target 고정 (보조 진단)"
        lines += [f"## 2. 순위 민감도 S — {name}\n", "```\n" + t.round(3).to_string() + "\n```\n"]
        D = d.pivot_table(index=["cond", "metric", "scope", "seed"], columns="model", values="S").reset_index()
        D["D"] = D.shortcut - D.clean
        Dt = D[D.scope.isin(["all", "aligned", "conflicting"])].groupby(["metric", "scope", "cond"]).D.agg(
            mean="mean", pos_seeds=lambda v: int((v > 0).sum())).unstack("cond")
        lines += [f"### D = S_shortcut − S_clean — {name}\n", "```\n" + Dt.round(3).to_string() + "\n```\n"]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    make_fig(sv)


def make_fig(sv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"clean": "#2a78d6", "shortcut": "#eb6834"}
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                         "axes.edgecolor": "#e4e3df", "font.size": 9, "legend.frameon": False})
    d = sv[(sv["eval"] == "new_attr") | (sv.cond == "original")]
    panels = [(q, sc) for q in ["mif", "srg"] for sc in ["all", "aligned", "conflicting"]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for ax, (q, sc) in zip(axes.ravel(), panels):
        x = np.arange(len(CONDS))
        for k, model in enumerate(["clean", "shortcut"]):
            g = d[(d.metric == q) & (d.scope == sc) & (d.model == model)]
            m = [g[g.cond == c].S.mean() for c in CONDS]
            ax.bar(x + (k - 0.5) * 0.36, m, 0.32, color=C[model], label=model, zorder=2)
            for i, c in enumerate(CONDS):
                pts = g[g.cond == c].S.values
                ax.scatter(np.full(len(pts), x[i] + (k - 0.5) * 0.36), pts, s=10, color="#0b0b0b", zorder=3)
        ax.set_xticks(x, CONDS)
        ax.set_title(f"{q.upper()} · {sc}")
    axes[0, 0].set_ylabel("rank sensitivity S")
    axes[1, 0].set_ylabel("rank sensitivity S")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2)
    fig.suptitle("ResNet-18: rank sensitivity after extra training (new attributions; bar = seed mean, dot = seed)",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "figs" / "fig_finetune_sensitivity.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
