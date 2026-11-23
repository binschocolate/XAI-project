"""Baseline OOD 정량화 결과 요약표 + 그림 (CPU). 입력: outputs/baseline_ood/<run>/ood.csv

출력 (results/baseline_ood/)
  ood_all.csv                 13개 run 의 ood.csv 를 합친 것 (+ arch, model 열)
  summary_fraction025.csv     25% 제거, scope=all 의 모델 유형별 평균 (OOD AUROC·정보 잔존)
  figs/fig_ood_vs_retention.png   25% 제거: 정보 잔존(cosine) vs OOD AUROC(Mahalanobis), baseline 별
  figs/fig_ood_by_fraction.png    제거 비율별 OOD AUROC (Mahalanobis), 모델 유형별
  figs/fig_ptarget_by_group.png   25% 제거: 원래 예측 class 확률 (aligned / conflicting, clean vs shortcut)
사용: python src/plot_baseline_ood.py
"""
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "outputs" / "baseline_ood"
OUT = REPO / "results" / "baseline_ood"
BASES = ["black", "mean", "blur", "inpaint", "trainset"]
BASE_N = {"black": "Black", "mean": "Mean", "blur": "Blur", "inpaint": "Inpaint", "trainset": "Train-set"}
BASE_C = {"black": "#2a78d6", "mean": "#eb6834", "blur": "#1baf7a", "inpaint": "#eda100", "trainset": "#e87ba4"}
MODEL_C = {"clean": "#2a78d6", "shortcut": "#eb6834"}
TYPES = [("IN-R50", "ref"), ("R18", "clean"), ("R18", "shortcut"), ("R50", "clean"), ("R50", "shortcut")]
TYPE_N = {("IN-R50", "ref"): "ImageNet R50 (ref)", ("R18", "clean"): "R18 clean", ("R18", "shortcut"): "R18 shortcut",
          ("R50", "clean"): "R50 clean", ("R50", "shortcut"): "R50 shortcut"}
MARK = {("IN-R50", "ref"): "*", ("R18", "clean"): "o", ("R18", "shortcut"): "^", ("R50", "clean"): "s",
        ("R50", "shortcut"): "D"}
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                     "axes.edgecolor": "#e4e3df", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def load():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(IN / "*" / "ood.csv")))])
    d["arch"] = d.run.str.split("_").str[0].replace({"resnet18": "R18", "resnet50": "R50", "imagenet": "IN-R50"})
    d["model"] = np.where(d.run.str.contains("rho095"), "shortcut", np.where(d.run.str.contains("rho050"), "clean", "ref"))
    return d


def fig_scatter(d):
    a = d[(d.scope == "all") & (d.fraction == 0.25) & d.baseline.isin(BASES)]
    m = a.groupby(["arch", "model", "baseline"])[["cos_to_orig", "auroc_maha"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for t in TYPES:
        for b in BASES:
            r = m[(m.arch == t[0]) & (m.model == t[1]) & (m.baseline == b)]
            ax.scatter(r.cos_to_orig, r.auroc_maha, color=BASE_C[b], marker=MARK[t], s=70 if t[1] != "ref" else 140,
                       edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
    for b in BASES:  # baseline 이름은 그 baseline 점들의 평균 위치 옆에 직접 표기
        r = m[m.baseline == b]
        ax.annotate(BASE_N[b], (r.cos_to_orig.mean(), r.auroc_maha.mean()), xytext=(10, -14), textcoords="offset points",
                    color=INK, fontsize=9)
    hs = [plt.Line2D([], [], color=BASE_C[b], marker="o", ls="", label=BASE_N[b]) for b in BASES] + \
         [plt.Line2D([], [], color=INK2, marker=MARK[t], ls="", label=TYPE_N[t]) for t in TYPES]
    ax.legend(handles=hs, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(ax.get_xlim()[0], 0.51, " 0.5 = indistinguishable from original", color=INK2, fontsize=7.5, va="bottom")
    ax.set_xlabel("information retained: cosine(masked, original features)  →")
    ax.set_ylabel("OOD-ness: AUROC original vs masked (Mahalanobis)  →")
    ax.set_title("Masking baselines at 25% random-patch removal (scope = all, seed mean)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "fig_ood_vs_retention.png", dpi=140)
    plt.close(fig)


def fig_fraction(d):
    a = d[(d.scope == "all") & d.baseline.isin(BASES)]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
    for ax, t in zip(axes, TYPES):
        g = a[(a.arch == t[0]) & (a.model == t[1])].groupby(["baseline", "fraction"]).auroc_maha.mean().unstack()
        for b in BASES:
            ax.plot(g.columns, g.loc[b], color=BASE_C[b], lw=2, marker="o", ms=5, label=BASE_N[b])
        ax.set_title(TYPE_N[t])
        ax.set_xticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"])
        ax.set_xlabel("patches removed")
    axes[0].set_ylabel("OOD AUROC (Mahalanobis)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=5)
    fig.suptitle("OOD-ness by removal fraction (seed mean)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "figs" / "fig_ood_by_fraction.png", dpi=140)
    plt.close(fig)


def fig_group(d):
    a = d[(d.scope.isin(["aligned", "conflicting"])) & (d.fraction.isin([0.0, 0.25])) & (d.model != "ref")]
    m = a.groupby(["arch", "model", "scope", "baseline"]).p_target.mean()
    cols = ["original"] + BASES
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharey=True)
    xs = np.arange(len(cols))
    for r, arch in enumerate(["R18", "R50"]):
        for c, sc in enumerate(["aligned", "conflicting"]):
            ax = axes[r, c]
            for k, model in enumerate(["clean", "shortcut"]):
                v = [m[(arch, model, sc, b)] for b in cols]
                ax.bar(xs + (k - 0.5) * 0.36, v, 0.32, color=MODEL_C[model], label=model, zorder=2)
            ax.set_xticks(xs, ["Original\n(no mask)"] + [BASE_N[b] for b in BASES])
            ax.set_title(f"{arch} · {sc} ({'background matches class' if sc == 'aligned' else 'background ≠ class'})")
            if c == 0:
                ax.set_ylabel("prob. of original predicted class")
            ax.set_ylim(0, 1.05)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2)
    fig.suptitle("25% random-patch removal: prediction retention by group (seed mean)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "figs" / "fig_ptarget_by_group.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False
    d = load()
    d.to_csv(OUT / "ood_all.csv", index=False)
    s = d[(d.scope == "all") & (d.fraction == 0.25)].groupby(["arch", "model", "baseline"])[
        ["auroc_maha", "auroc_knn", "cos_to_orig", "pred_agree", "p_target"]].mean().reset_index()
    s.to_csv(OUT / "summary_fraction025.csv", index=False)
    fig_scatter(d)
    fig_fraction(d)
    fig_group(d)
    print("saved ->", OUT)
