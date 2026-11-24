"""ROAR 결과 분석 (CPU). 입력: outputs/roar/<run>/roar.csv, outputs/analysis/faith_auc.csv(MIF 비교용)

지표
  - ROAR 점수 = 지우는 비율 10/30/50/70% 에서의 test 정확도 평균 (낮을수록 중요한 곳을 잘 찾은 설명)
  - 정확도 하락 = 지우기 0% 재학습 정확도 − ROAR 점수
  - Random 대비: 설명 방법의 ROAR 점수가 Random 보다 낮은지 (아니면 그 설명은 무작위보다 나을 게 없음)
  - baseline 간 순위 변화: Grad-CAM·IG·RISE 순위를 mean / trainset 에서 비교 (Kendall τ-b, 역전 여부)
      → shortcut vs clean (같은 arch·seed) 비교: D_roar = (1−τ)_shortcut − (1−τ)_clean
  - Deletion(MIF) 과의 일치: 같은 baseline 에서 ROAR 순위 vs MIF 순위 Kendall τ-b
출력 (results/roar/): roar_all.csv, roar_scores.csv, ranks.csv, summary.md, figs/fig_roar_curves_<arch>.png
사용: python src/roar_analysis.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "roar"
XAI = ["gradcam", "ig", "rise"]
METHODS = XAI + ["random"]
BASES = ["mean", "trainset"]
MC = {"gradcam": "#2a78d6", "ig": "#eb6834", "rise": "#1baf7a", "random": "#eda100"}
MN = {"gradcam": "Grad-CAM", "ig": "IG", "rise": "RISE", "random": "Random"}


def tau(a, b):
    t = kendalltau(a, b)[0]  # τ-b (동점 처리), 정의 불가면 NaN
    return t


def main():
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(REPO / "outputs" / "roar" / "*" / "roar.csv")))])
    d.to_csv(OUT / "roar_all.csv", index=False)
    base = d[d.method == "none"].set_index("run").test_acc
    s = d[d.method != "none"].groupby(["run", "arch", "model", "seed", "method", "baseline"]).test_acc.mean() \
        .rename("roar_score").reset_index()
    s["acc_no_removal"] = s.run.map(base)
    s["drop"] = s.acc_no_removal - s.roar_score
    rnd = s[s.method == "random"].set_index(["run", "baseline"]).roar_score
    s["better_than_random"] = s.roar_score < s.set_index(["run", "baseline"]).index.map(rnd)
    s.to_csv(OUT / "roar_scores.csv", index=False)

    # 순위: baseline 별 (낮은 점수가 좋음)
    fa = pd.read_csv(REPO / "outputs" / "analysis" / "faith_auc.csv")
    fa = fa[(fa.scope == "all") & fa.method.isin(XAI)]
    rows = []
    for run, g in s[s.method.isin(XAI)].groupby("run"):
        sc = {b: g[g.baseline == b].set_index("method").loc[XAI].roar_score.values for b in BASES}
        r = dict(run=run, arch=g.arch.iloc[0], model=g.model.iloc[0], seed=g.seed.iloc[0],
                 tau_mean_vs_trainset=tau(sc["mean"], sc["trainset"]))
        for b in BASES:
            r[f"rank_{b}"] = ">".join(np.array(XAI)[np.argsort(sc[b], kind="stable")])
            mif = fa[(fa.run == run) & (fa.baseline == b)].set_index("method").loc[XAI].mif.values
            r[f"tau_roar_vs_mif_{b}"] = tau(sc[b], mif)  # 둘 다 낮을수록 좋음 → 같은 방향
        r["flip_mean_vs_trainset"] = r["rank_mean"] != r["rank_trainset"]
        rows.append(r)
    rk = pd.DataFrame(rows)
    rk["S_roar"] = 1 - rk.tau_mean_vs_trainset
    rk.to_csv(OUT / "ranks.csv", index=False)
    piv = rk.pivot_table(index=["arch", "seed"], columns="model", values="S_roar")
    piv["D_roar"] = piv.shortcut - piv.clean

    pd.set_option("display.width", 200)
    lines = ["# ROAR 결과 요약\n",
             "## 1. ROAR 점수 (지운 비율 10–70% 평균 test 정확도, 낮을수록 좋음; seed 평균)\n",
             "```\n" + s.pivot_table(index=["arch", "model", "baseline"], columns="method", values="roar_score")
             [METHODS].round(3).to_string() + "\n```\n",
             "지우기 0% 재학습 정확도 (seed 평균):\n",
             "```\n" + d[d.method == "none"].groupby(["arch", "model"]).test_acc.mean().round(3).to_string() + "\n```\n",
             "## 2. Random 보다 나은 비율 (run 수 기준)\n",
             "```\n" + s[s.method.isin(XAI)].groupby(["arch", "baseline", "method"]).better_than_random.mean()
             .unstack("method").round(2).to_string() + "\n```\n",
             "## 3. baseline(mean vs trainset) 에 따른 순위 변화\n",
             "```\n" + rk[["run", "rank_mean", "rank_trainset", "flip_mean_vs_trainset", "tau_mean_vs_trainset"]]
             .to_string(index=False) + "\n```\n",
             "S_roar = 1 − τ(mean, trainset), D_roar = shortcut − clean (같은 arch·seed):\n",
             "```\n" + piv.round(3).to_string() + "\n```\n",
             "## 4. Deletion(MIF) 순위와의 일치 (Kendall τ-b, 1 = 같은 순위)\n",
             "```\n" + rk.groupby(["arch", "model"])[["tau_roar_vs_mif_mean", "tau_roar_vs_mif_trainset"]].mean()
             .round(3).to_string() + "\n```\n"]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    make_fig(d)


def make_fig(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                         "axes.edgecolor": "#e4e3df", "font.size": 9, "legend.frameon": False})
    for arch in sorted(d.arch.unique()):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey=True)
        for r, model in enumerate(["clean", "shortcut"]):
            dm = d[(d.arch == arch) & (d.model == model)]
            a0 = dm[dm.method == "none"].test_acc.mean()
            for c, b in enumerate(BASES):
                ax = axes[r, c]
                for m in METHODS:
                    g = dm[(dm.method == m) & (dm.baseline == b)].groupby("fraction").test_acc.mean()
                    ax.plot([0] + list(g.index), [a0] + list(g.values), color=MC[m], lw=2, marker="o", ms=4,
                            ls="-" if m != "random" else (0, (3, 2)), label=MN[m])
                ax.set_title(f"{model} · fill = {b}")
                if r == 1:
                    ax.set_xlabel("fraction of patches removed (train and test)")
                if c == 0:
                    ax.set_ylabel("test accuracy after retraining")
        name = {"resnet18": "ResNet-18", "resnet50": "ResNet-50"}[arch]
        fig.suptitle(f"{name}: ROAR curves (lower = more important regions removed; seed mean)", x=0.01, y=0.99, ha="left")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=4)
        fig.tight_layout(rect=(0, 0, 1, 0.89))
        fig.savefig(OUT / "figs" / f"fig_roar_curves_{arch}.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
